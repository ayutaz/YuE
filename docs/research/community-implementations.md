# コミュニティ実装の調査 — 実装マップと実測値の読み方

調査日 2026-09-18。YuE2は2026年9月10日公開で、この時点でコミュニティ実装は**約1週間の蓄積**しかない。

## 0. 全体像 — 「部品は揃っているが、機能は誰も完成させていない」

| コミュニティにあるもの | 実体 | 「参照音声で声を指定」に使えるか |
|---|---|---|
| 音声→semantic token エンコーダ(tokenizer head v4〜v9) | **部品**。公式の欠落を埋める | 単体では不可。学習かICLの前提条件 |
| 話者ごとのLoRA学習ワークフロー(Voices / Trainer) | **ワークフローは存在**するが推論時に参照音声を渡すのではなく**話者ごとに学習が必要** | 効果が測定上ノイズ範囲(§3) |
| 推論時に参照音声をNARへ注入する機構(hum-to-song) | **機構は実証済み** | **音色を意図的に捨てる設計**。声には使われていない |
| 汎用トレーナーへの統合(ai-toolkit) | 正式merge済み | 学習基盤。声の条件付け機構は含まない |

→ 独自実装の性質は「既存機能の改善」ではなく **未完成部分の実装**。

## 1. 実装4系統の詳細

### 1.1 Mothersuperior realaudio tokenizer(最も本格的・他実装の基盤)

- [HF: yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)
- 解説記事: [ComfyUI Wiki: YuE2 Real-Audio Encoder](https://comfyui-wiki.com/en/news/2026-09-16-yue2-realaudio-encoder)
- 作者: Reddit `thatisnotmychapstick` / HF `Mothersuperior`。2026年9月公開

**構成**

| 成果物 | 内容 |
|---|---|
| tokenizer head | **MERT-v2-FullSong layer-20 特徴(25Hz、instance-normalized)→ 32,768 semantic codes**。8層 transformer encoder、d=512、512フレーム窓 |
| NAR LoRA | **rank-32**。`nar_self_attn.{q,k,v,o}_proj` と `nar_mlp.{gate,up,down}_proj` × 28層 + `vae2llm` / `llm2vae` は**フル置換重み** |
| スクリプト | `prep_real.py`(MERT特徴+VAE潜在抽出)、`cursor_prep.py`(Demucs+MMSで歌詞アライン)、`joint.py`(head/decoderの自前音声適応)、`ar_prep.py`(トークン化+regularizer混合) |
| regularizer | `Mothersuperior/yue2-minted-corpus`(YuE2生成曲 4,732曲)。別リポジトリとして必要 |

**学習の仕組み(自己教師あり)**: YuE2の生成曲は「音声と正解トークンのペア」になるため、まずそれで head を学習。
実録音への適応は **YuE2自身のデコーダを採点器として使う**(手動ラベル付け不要)。
v5の再構成を Kytra が聴いたことから「レンダリング音声を実録音に対してペナルティする」案が生まれた。

**精度の推移**

| バージョン | top-1 一致(held-out YuE2曲) | 備考 |
|---|---|---|
| v4 | **16.1%** | legacy |
| v5 | top-1 **18.9%** / top-5 **45.0%** / top-1-or-neighbour **34.1%** | フルコーパスで再学習 |
| v8, v9 | 同程度のtoken精度で音声再構成の忠実度が改善 | **audio-domain lossで学習。v9が現行** |

聴感評価は v7 > v6、v8 > v7、総合で **v9が最良**。NAR round-trip の聴感は約95%。
HFリポジトリには v4 / v5 / v8 / v9 が存在(v6, v7は非公開)。全体 2.87GB。

**自前アーティストAR LoRAの手順**

1. `prep_real.py` — MERT特徴 + VAE潜在
2. `cursor_prep.py` — Demucs + MMS で歌詞アライン
3. `joint.py` — head/decoder の自前音声適応(任意)
4. `ar_prep.py` — トークン化 + regularizer pack 併合
5. **rank-64 AR LoRA** を「アーティスト:minted = 50/50、lyric-cursor weight 0.08」で 3,000ステップ、
   step 600以降 200ステップごとにチェックポイント。**著者注記: 約1,500ステップを超えて学習するな(記憶化する)**

**必要データ(曲ごと)**: `.flac` + セクション記号付きフル歌詞 `.lyrics.txt` + トリガー語入りスタイルキャプション `.txt`

**既知の残課題**: トークン選択誤りによる**音程外れ**がデコーダ学習では解消しない。
mel spectrogram loss は知覚品質と相関せず、**LTAS(long-term average spectrum)距離**の方が信頼できた。

### 1.2 ComfyUI-YuE2-Voices + YuE2-Voices-training-kit(「歌声の再利用」を明示的に狙う唯一の実装)

- [GitHub: KadoBOT/ComfyUI-YuE2-Voices](https://github.com/KadoBOT/ComfyUI-YuE2-Voices)
- [HF: rambrogi/YuE2-Voices-training-kit](https://huggingface.co/rambrogi/YuE2-Voices-training-kit)

**ノード構成**: `Extract Voice`(RoFormerでボーカル/伴奏分離)→ `Train YuE2 Voice` → `Apply YuE2 Voice` / `Apply YuE2 Decoder` / `Mix Voice Layers`

**学習対象**: **AR分岐のみ**。ソース音声が生成したトークンに対する **token cross-entropy**。
kohya形式のLoRAで **112個のmerged projection**(qkv/o/gate_up/down)を対象、
隔離されたPythonワーカー内で **23GiBのアロケータ上限**付きで実行。

> 「歌手類似度も音響再構成も直接最適化していないので、artist lossが低いことは
> **アダプタがソースのsemantic tokenを予測できる**ことを意味するだけで、
> 未知の歌詞がソースの声で出てくることを意味しない」(README)

**ARがどう音響側に届くか**: ComfyUIはprefix+サンプル済みトークンでARのprefillを再実行し、
**全層のkey/value statesを音響transformerに条件として渡す**(`comfy/text_encoders/yue2.py::_acoustic_conditioning`)。
→ トークンIDが同一でも、AR重みが変わればレンダリングは変わる。

**training-kitの3点セット**(headとdecoderは**必ずペアで使用**)

| 成果物 | サイズ | 内容 |
|---|---:|---|
| tokenizer head | 86MB | ARボイストレーナー用にsemantic tokenを読み出す |
| NAR decoder | 108MB | トークンを音響側で音声に戻す。**実録音潜在に対する flow-matching loss で学習** = 音色・発音・フレージングを形作る |
| minted regularizer | 102MB | 4,516曲 + holdout。アーティスト同一性の記憶化を防ぐ |

配置は `ComfyUI/models/audio/yue2-voices/{tokenizer,minted/regularizer}/`。

**正当性の検証**(READMEの Correctness notes):
ワーカーのバックボーンはComfyUIのYuE2 AR forwardの行単位ミラーで、`worker/verify_backbone.py` が
ネイティブチェックポイントに対して **bit-exact(差分0.0)なロジット**を検証。
ボイスアダプタは CLIP分岐で **112/112**、NAR decoderは MODEL分岐で **114/114** のターゲットを解決(未ロードキー警告なし)。

### 1.3 ComfyUI-YuE2-Trainer(最も導入が容易)

- [GitHub: Starnodes2024/ComfyUI-YuE2-Trainer](https://github.com/Starnodes2024/ComfyUI-YuE2-Trainer) / [fork: Saganaki22](https://github.com/Saganaki22/ComfyUI-YuE2-Trainer)

**学習対象**: **NAR分岐のみ**、released flow-matching objective。ARは凍結。

| 項目 | 値 |
|---|---|
| rank / alpha | 32 / 32 |
| steps | 既定3,000(1,000〜5,000を探索) |
| learning rate | 1e-4 |
| EMA decay | 0.999(既定で有効) |
| scheduler | cosine |
| データ | 5〜30曲、一貫したスタイル。キャプション `.txt` は任意(トリガー語でも可) |
| クリップ長 | 既定10秒(`clip_seconds`) |
| 形式 | MP3 / WAV / FLAC |
| VRAM | **24GB推奨**(RTX 5090 Laptopで検証)。不足時は `clip_seconds` 6〜8 か `adamw_8bit` |

**条件付けは text-only(codec dropout regime)**: チェックポイント固有のテキスト接頭辞
(`[Tags] your_trigger_word, caption ...`)で条件付けし、音声は内蔵VAEで潜在化(30秒チャンク、ディスクキャッシュ)。
外部エンコーダ不要。理由は README が明言する通り
「**任意音声の正解semantic tokenはm-a-pから提供されていない**」ため。

**著者自身の限界表明**

> AR「作曲」分岐は凍結される(m-a-pは audio→token エンコーダも学習コードも公開していない)ので、
> これは **style/timbre LoRA であり、完全なボイスクローンではない**
> 「Voice cloning still dont work」
> 「これは録音潜在の再構成を最適化するものであり、モデルがアーティストのスタイルや声を再現することを立証しない」
> 「公式のYuE2学習コードは存在しない。ここでの学習目標は公開された推論コードから再構成したものである」

### 1.4 YuE2-hum-to-song(参照音声を推論時にNARへ注入する機構の実例)

- [HF: Mothersuperior/YuE2-hum-to-song](https://huggingface.co/Mothersuperior/YuE2-hum-to-song)

**2段構成**

1. **AR planner(学習なし)**: 鼻歌を SheetSage2 でABCに転写し、`[ABC_START] + hummed_bars` を
   **閉じトークンなしの開放プレフィックス**として与える。プランナーがその続きとしてフルスコアを書く。
2. **NAR decoder(学習済みLoRA)**: 元音声のピッチを追跡し、**サイン波として再合成**してVAEで潜在化(= carrier latent)。
   **zero-init projection 4本**で、デコーダのhidden stateに **入力時と layer 7 / 14 / 21 の前で加算注入**。

**学習**: 分離ボーカルからの**自己教師あり 3,163ペア**(condition = sine carrier / target = full mix)。

**音色**: **意図的に捨てている**。「モデルはサイン波表現しか見ない」ため、
実ボーカルステムで作った学習セットが「ラップトップのマイクに鼻歌を吹き込む人」に汎化する。
出力はYuE2が学習した声の特性であり、入力話者の声ではない。

**効果**: 控えめ。precision 0.960 → **0.906**(hum有無)。著者の説明は
「**文脈のsemantic tokenが既にメロディを固定しているので、carrierはタイミングとピッチのニュアンスしか足せない**」。

→ **この機構(zero-init注入)をサイン波ではなく対象歌手の実ボーカル潜在に転用するのが、独自実装の本命**。
詳細は [custom-training-plan.md](custom-training-plan.md) §3。

### 1.5 ostris/ai-toolkit(汎用トレーナーへの正式統合)

- [PR #1042 "YuE2" by jaretburkett](https://github.com/ostris/ai-toolkit/pull/1042) — **2026-09-14 merge済み**、12コミット
- 追加内容: YuE2の実験的サポート、duration selector、caption/metadataオプション、
  **譜面入力を50%ドロップアウトして学習**(譜面あり/なし両方を扱えるようにする)、dataset dropout設定、正則化パラメータ
- tokenizerは §1.1 のもの(謝辞「special thanks to Kytra for the tokenizer, without which, this wouldn't be possible」)
- 運用指針([RunComfy解説](https://www.runcomfy.com/trainer/ai-toolkit/yue2-lora-training)):
  **AR expertは速く記憶し、NAR expertがスタイルを描く**。多様な曲、セクション付きフル歌詞で**caption dropoutを0**、
  **KLアンカリング**、ABC計画のmixed-mode dropout、**潜在キャッシュ必須**、未学習の歌詞での早期チェック。
  「画像LoRAのレシピに音声を差し替えたものではなく、付属のYuE2プリセットから始めること」

### 1.6 周辺エコシステム([github.com/topics/yue2](https://github.com/topics/yue2))

| リポジトリ | 内容 |
|---|---|
| `remiqora`(41★) | ACE-Step 1.5 + YuE2-3B のローカル音楽スタジオ。LoRA微調整、MIDI転写、マルチトラックDAW |
| `YuE2-Studio` | クラウドワークステーション。**RVC / Seed-VC のデュアルモード歌声変換**を後処理に併用。24GB GPU向け1スクリプト配備 |
| `audio.cpp`(2.8k★) | C++推論エンジン。voice conversion / music generation、Python非依存、AMD GPU / Apple Silicon対応 |
| `yue2-studio` | Apple Silicon向けmacOSアプリ(MLX)。カバーと転写 |
| `yue2-concept-sliders` | 16種のYuE2音楽コンセプトスライダー(ComfyUI) |
| `KytraScript/ComfyUI-FS_Audio_Suite` | モジュール式YuE2音声生成ノード群 |

## 2. 自己申告されている欠陥(= 独自実装の伸び代)

ComfyUI-YuE2-Voices のREADMEが自ら挙げている問題。**これが「negative resultは不可能の証明ではない」根拠**。

1. **AR trainerが各シーケンスの先頭ウィンドウしかlossに通していなかった**
   > 4曲ランは確かに多くの音声を供給した(330秒 対 3クリップ約77秒)が、ARトレーナーは各シーケンスの
   > 開始ウィンドウのみをlossに通していたので、どちらでもほぼ同じ素材しか学習に届いていない。
   > **あの数字は供給した音声量であって、lossが見た音声量ではない。full temporal coverageで再実行してから
   > 比較を読むべき**
2. **検証スクリプトの2つのバグ**(修正済み、**旧結論は撤回**)
   - 生のtokenizer-head indexを、絶対語彙IDを期待する条件付けに渡していた(codec offset未適用 → 誤ったトークン範囲)
   - 48kHzデコーダの出力を44,100Hzで書き出していた
   - → 「held-out vocalが自身のトークンに対して0.585、全アダプタ組合せが約0.0015(=偶然)」という旧測定と、
     「ARアダプタは新曲に同一性を運べない」という主張は、**結果ではなく未確立として扱うべき**と著者が明記
3. **データ量が極小**(77〜330秒)
4. **指標が粗く、単一ラン**(§3.4)

## 3. 実測値 — 3つの独立した測定を混同しないこと

### 3.1 (a) アダプタ有無の比較 — `checks/mert_identity.py`

held-out ボーカルに対する MERT cosine:

| 対象 | スコア |
|---|---:|
| 参照(自己対照) | **1.000** |
| **stock model(素のモデル)** | **0.852** |
| 3クリップ adapter | 0.846 |
| 4曲 adapter | 0.847 |

→ **アダプタは素のモデルをわずかに下回る**。
**ただし暫定値**: スコアラが全クリップを参照の正確なサンプル数に合わせており、time-warpとピッチずれが生じている。

### 3.2 (b) トークン列を固定した参照再構成 — `checks/compare_reconstruction.py`

アーティスト適応済みheadがheld-outボーカルから読み出した**1本のトークン列を固定**し、
各アダプタ組合せでレンダリングしてhold-outボーカルと比較(トークン選択を統制するため):

| Render | vs held-out vocal (mel / MFCC / envelope) |
|---|---|
| stock(両アダプタなし) | 0.120 / 0.019 / 0.058 |
| voice adapter のみ | 0.108 / −0.030 / 0.105 |
| NAR decoder のみ | 0.098 / −0.121 / −0.016 |
| both | 0.095 / −0.104 / 0.002 |

pairwise では **voice adapterはレンダリングをほとんど動かさない**(mel 0.983 / MFCC 0.996)一方、
NAR decoderはより大きく動かす(mel 0.923 / envelope 0.541)。
→ **どちらも held-out 演奏の方向へは動かない**。

### 3.3 (c) デコーダ選択の比較 — 変えているのは「どのデコーダを積むか」だけ

2曲フル、歌詞のvocal tagをバイト単位で固定しジャンルのみ変更、対象アーティストの実ボーカルとのMERT cosine。
**表の軸は `Decoder` であり、ARボイスアダプタは全行で適用されている**(2行目が "AR-conditioned"):

| Decoder | vs アーティスト | per-voiceとのpairwise |
|---|---:|---:|
| stock decoder(NARアダプタなし) | **0.8880** | 0.9997 |
| per-voice NAR (AR-conditioned) | 0.8851 | 1.0000 |
| generic NAR | 0.8649 | 0.9994 |

著者の読み:

> 2つのper-voice経路は互いに0.9997similarなので、音響側を話者ごとに学習することは
> **どのデコーダをロードするか**よりレンダリングを動かさない。genericデコーダは両者より約0.02低い。
> アーティストの性格は再現される(**0.885〜0.888 対 自己対照1.000**、しかもアダプタが学習していないジャンルで)
> ので声自体は偶然ではないが、**それを運んでいるのはNAR decoderの差し替えではない**。

なお per-voice NAR の学習自体は改善している(flow loss 0.987 → 0.832、修正前は 0.938 → 0.857)が、
**レンダリングはstockから分離しなかった**。

### 3.4 指標の読み方(重要)

著者自身の注記:

> MERT cosineは粗い表現であり、**表内の相対比較のみが統制されている。絶対値は同一性の百分率ではない**。
> per-voice行は単一の学習ランなので、**stockとの0.003差は単一シードが動かす範囲内**。

**0.888は「88.8%本人」ではない。** MERTは音楽内容(ジャンル・編成・ミックス)を符号化する表現なので、
同系統の曲どうしなら別人でも0.85〜0.9に乗る。
→ 独自評価では **話者照合用の埋め込み**(例: `pyannote/wespeaker-voxceleb-resnet34-LM`)で測り直すべき。
歌唱は話者照合モデルの想定外条件なので、**無伴奏ボーカルに分離してから測る**、
**同一話者の別曲ペアで閾値を校正する**といった前処理が必要。

## 4. プロンプト側の要因は測定可能な差を出している

**声を実際に動かしているのはプロンプトだった**という、最も実用的な発見。

| 操作 | 効果 |
|---|---|
| 歌詞から delivery tag(`[Growling, strained baritone]`、`[SCREAMING, gritty baritone]` 等)を全除去 | per-voiceレンダリングが **0.909 → 0.915**。stockとper-voice decoderの差が **0.0034 → 0.0001** に収縮 |
| ジャンルを一致させる vs 無関係なスタイル | **0.909 vs 0.885**。「大きなスタイル変更は、どのデコーダ選択よりも声を動かす」 |

その他の運用ガイダンス(README「Keeping a voice stable across generations」):

- **両方のApplyノードを繋ぐ**。ボイスアダプタ単独は「AIが自分のボーカルを変えてしまった」の最大の原因
- 1声につきApplyノード1つ、strength約1.0。**同じアダプタを2回積む(約1.5倍)と音色が破綻する**
- music段のサンプリングを締める: **temperature 0.7〜0.85 / top_k 30〜50 / top_p 約0.9**。
  緩いサンプリング(temp 1.0, top_k 100)は演奏と音素をドリフトさせる
- **repetition penaltyは1.05〜1.1付近**に。penaltyは出現回数で累乗される(`penalty ** counts`)ため、
  数字の見た目よりはるかに強く効き、高い値はモデルに音素を回避させ単語を書き換えさせる
- `max_duration` を歌詞の長さに合わせる。20秒分の歌詞に60秒窓を与えると、捏造されたad-libで埋まる
- **歌われる語を正確にタイプし、声の描写はプロンプトから外す**。声の記述はアダプタと競合する
- 再現にはシードを両方固定。ABCスコアがメロディを固定するので、同じABCから再生成すれば音素タイミングを保てる
- **学習は小刻みに(300〜600ステップ)**。artist lossが下がる一方でminted-valが平坦なことを見る。
  過学習はテイクを記憶し、新しい歌詞が崩れる

## 5. ライセンス(コミュニティ実装共通)

- YuE2の重みは **CC BY-NC 4.0** → **そこから学習したLoRA/アダプタは派生物であり、同じ非商用条件が適用される**。
  調査した実装はいずれもこの点を明記している。
- 各実装のコード自体は別ライセンス(ComfyUI-YuE2-Trainerは **MIT**)。
- 詳細と個人利用での扱いは [custom-training-plan.md](custom-training-plan.md) §8。
