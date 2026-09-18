# 独自実装・学習の投資対効果と実行計画

前提: **YuE2単独で完結**(後段の歌声変換を使わない)/ 予算 **$500〜1000** / **個人利用** / 日本語対応も狙う。
調査日 2026-09-18。

> **⚠ §5 のフェーズ計画と予算は [implementation-milestones.md](implementation-milestones.md) の
> M0〜M6 に置き換わっています。** 本文書は「独自実装に意味があるか」の**投資判断の根拠**として残し、
> 実行計画はマイルストーン文書を参照してください。置き換わった理由は2つ:
>
> 1. 対象話者データが **4〜6曲・約10分で固定**と判明し、設計方針が変わった
>    ([lora-architecture-and-data-scale.md](lora-architecture-and-data-scale.md) §4.7)
> 2. **ローカルの RTX 4090(24GB)を主環境**にしたため、総額が $430〜1,010 → **$90〜470** に下がった
>    ([implementation-milestones.md](implementation-milestones.md) の実行環境節)
>
> ### 旧フェーズ名 → 現マイルストーンの対応
>
> | 本文書 (§5) | [lora-...-data-scale.md](lora-architecture-and-data-scale.md) §5.2 | [implementation-milestones.md](implementation-milestones.md) |
> |---|---|---|
> | Phase 0(評価ハーネス) | G0 | **M0** |
> | Phase 0.5(参照音声ICL注入) | G0.5 | **M2** |
> | Phase 1(tokenizer head round-trip) | G0.5 に統合 | **M2** の完了条件2 |
> | Phase 2(NAR注入アダプタ学習) | A | **M3** |
> | Phase 4(per-voice 微調整) | B | **M4** |
> | Phase 3(日本語 diction) | C | **M6** |
> | —(新設) | — | **M1** 学習基盤の実装 / **M5** operating envelope の実測 |

## 1. 結論

**「コミュニティ実装をなぞる」だけでは目標に到達しない。一方で独自実装に意味がある領域が3つ明確に存在する。**
ただし独自実装の性質は「既存機能の改善」ではなく **未完成部分の実装**であり、増分改善より成功確率は低い。

| 選択肢 | 判断 | 根拠 |
|---|---|---|
| 素のモデルに参照音声を渡す | **✗ 機能が存在しない** | `SongRequest` に該当フィールド無し、音声→tokenエンコーダも未リリース |
| コミュニティ実装をそのまま使う | **✗ 目標未達** | MERT identity で stock 0.852 > adapter 0.846 / 0.847 |
| 既存実装の欠陥修正 + データ増 | **◎ 最優先** | AR trainerが各シーケンスの先頭ウィンドウのみをlossに通す欠陥。実効データ量77〜330秒 |
| NARに音色条件付けの入力経路を新設 | **◎ 本命** | semantic tokenは約375bpsで音色を運べない。hum-to-songが同型の注入機構を実証済み |
| 参照音声ICL注入(低レベル関数直叩き) | **○ 先に試す** | 実装20行。$0〜50で決着。効けばNAR学習の大半が不要になる |
| 日本語 diction / PER 改善 | **◎ 空白地帯** | 公式ベンチに言語別内訳がなく、日本語PERは誰も測っていない |
| tokenizer head をゼロから作り直す | **✗ 非推奨** | 既に v4→v9 の7世代反復がある。$500〜1000では追いつけない |
| 汎用zero-shot多話者クローン化 | **✗ 非推奨** | データ規模が2桁不足。個人利用なら単一話者に全振りが合理的 |

## 2. なぜ「なぞるだけ」では届かないのか

詳細は [community-implementations.md](community-implementations.md) §3。要点のみ:

- **(a) アダプタ有無**: 参照自己対照 1.000 / **stock 0.852** / 3クリップadapter 0.846 / 4曲adapter 0.847
  → アダプタが素のモデルを**下回る**
- **(b) トークン固定の参照再構成**: 「voice adapterはレンダリングをほとんど動かさない(pairwise mel 0.983 / MFCC 0.996)」
  「**どちらも held-out 演奏の方向へは動かない**」
- **(c) デコーダ選択**: stock decoder 0.8880 / per-voice 0.8851 / generic 0.8649、per-voice同士は pairwise 0.9997
  → 「**NAR decoderの差し替えは声を運ぶレバーではない**」
- **プロンプト側は効く**: delivery tag全除去で 0.909 → 0.915、ジャンル一致 0.909 vs 不一致 0.885
  → **現状「声」を決めているのはプロンプトと基盤モデルのpriorであり、LoRAではない**

**ただしこれは不可能の証明ではない**。著者自身が欠陥を列挙している(先頭ウィンドウのみのloss、
検証スクリプトのcodec offset/サンプルレート誤り(旧結論は撤回)、データ量77〜330秒、
指標が粗く単一ラン)。**未検証であって未達確定ではない。**

## 3. 本命 — hum-to-song型 zero-init 注入を「声」に転用する

### 3.1 なぜNAR側なのか

- semantic token は 32,768コード @25Hz ≒ **375 bps**。声の個性を運ぶ帯域がない
  ([yue2-voice-cloning-feasibility.md](yue2-voice-cloning-feasibility.md) §3.3)
- `nar_velocity()` で **NARはARの全hidden statesを参照**し、音色はNAR(flow matching)が
  AR hidden states と自身のpriorから生成している
- リクエストに話者を入れる口がない。**アーキテクチャに話者条件付けが存在しない**

→ 正攻法は **NAR分岐に音色を運ぶ入力経路を新設すること**。AR側にLoRAを当てる現行手法は隘路の手前で戦っている。

### 3.2 転用元の機構(既に実証・公開されている)

`Mothersuperior/YuE2-hum-to-song` の prosody adapter:

- 入力音声のピッチを追跡し、**サイン波として再合成**してVAEで潜在化(carrier latent)
- **zero-init projection 4本**で、NARデコーダのhidden stateに
  **入力時と layer 7 / 14 / 21 の前で加算注入**
- 分離ボーカルからの**自己教師あり 3,163ペア**(condition = sine carrier / target = full mix)で学習
- **音色は意図的に捨てている**(「モデルはサイン波表現しか見ない」)

### 3.3 転用案

**carrier をサイン波ではなく、対象歌手の実ボーカル潜在(別セグメント/別曲)にする。**

設計上の必須事項:

1. **cross-segment / cross-song ペアリング**。同一セグメントを条件に使うと
   「コピーの近道」を学習してしまい、話者同一性ではなく複写を覚える(zero-shot TTSの定石)
2. 条件側はボーカルのみ(分離ステム)、ターゲットはフルミックス。
   `jaCappella` は**最初からボーカルのみ**なので条件側データとしてそのまま使える
3. zero-init のまま開始し、注入強度をスケジュールする(既存の推論挙動を壊さない)
4. 注入層(入力 / 7 / 14 / 21)はそのまま踏襲し、まず前例の再現性を確認してから変更する
5. 汎化を狙わず**単一話者に全振り**する(個人利用なので合理的)

**根拠の強さ**: 機構・学習スクリプト・データ規模(3,163ペア)の前例があり、**予算内で再現可能なスケール**。
**リスク**: 前例は音色を運ばない設計で成功しており、「音色を運ばせる」方向での成功例は存在しない。

## 4. 参照音声ICL注入(Phase 0.5)

低レベル関数を直接叩けば約20行で書ける。詳細は
[yue2-voice-cloning-feasibility.md](yue2-voice-cloning-feasibility.md) §4 レベル3。

```python
ref_tokens = tokenizer_head(target_singer_audio)          # tokenizer head v9 が必要
prefix = plan.prefix + [t + CODEC_OFFSET for t in ref_tokens]
new_tokens, *_ = generate_tokens(model, prefix, sampling, seed, "semantic")
latents = nar_synthesize(model, prefix, ref_tokens + new_tokens, seed)
```

- **効く可能性**: YuE **v1** には dual-track ICL があった(分離vocal/accompanimentをトークンレベルで
  インターリーブし、20〜40秒セグメントを前置して**学習**していた)。公式も「ICLがLoRAに最も近い」と案内していた
- **効かない可能性**: YuE2ではAPIから消えており、**YuE2が codec-prefix ICL で学習されたかは非公開**。
  さらに375bpsの帯域制約があり、hum-to-songの著者も「文脈のsemantic tokenは既にメロディを固定しており、
  carrierはタイミングとピッチのニュアンスしか足せない」と述べている
- **なぜ先にやるか**: **$0〜50でどちらかに転ぶ**。効けばPhase 2の大半が不要、
  効かなければ「NARに口を作る」方針が実測で裏付けられる

## 5. フェーズ計画と予算(置き換え済み — 履歴として保持)

**この節は [implementation-milestones.md](implementation-milestones.md) の M0〜M6 に置き換わっています。**
GPU欄の「1×H100 80GB」「1×A100」は長窓設計前提の見積りで、
短窓設計ではローカル RTX 4090(24GB)で足ります(§6.1 の補足)。

| Phase | 内容 | GPU | 費用目安 |
|---|---|---|---:|
| **0** | 評価ハーネス構築(ASR 4パスPER + 話者類似度)。日本語4表記(漢字/ひらがな/カタカナ/ローマ字)のPER比較。stockの話者類似度ベースライン取得 | ローカル + 1×A100 数時間 | **$0〜30** |
| **0.5** | **参照音声ICL注入**(§4)。tokenizer head v9 で対象話者のトークンを取り、AR文脈にprependして声が乗るか判定 | 1×A100 5〜20h | **$0〜50** |
| **1** | tokenizer head **v9** で対象話者・日本語歌唱の round-trip 検証(top-k一致 + 聴感 + LTAS距離) | 1×A100 10〜20h | **$30〜80** |
| **2** | **本命**: NAR zero-init 注入アダプタを cross-segment ペアで自己教師あり学習(§3) | 1×H100 80GB 50〜100h | **$150〜350** |
| **3** | 日本語 diction AR LoRA(**full temporal coverage** 実装 + 日本語モーラアライメント) | 1×H100 50〜100h | **$150〜300** |
| **4** | 対象話者の per-voice 微調整 + アブレーション(タグ有無・ジャンル一致・複数シード) | 1×A100 50〜100h | **$100〜200** |
| | **合計(旧見積り)** | | **$430〜1,010** |
| | **現行見積り(M0〜M6)** | ローカル4090 + M5のみクラウド | **$90〜470** |

**価格前提**(2026年9月時点の調査):

| GPU | 価格帯 |
|---|---|
| A100 80GB | 約 **$0.60/h**(spot、Spheron)〜 **$1.07〜1.99/h**(on-demand)。オープン市場では$1/GPU-h以下も出ている |
| H100 SXM | **$1.49/h**(Vast.ai)〜 **$2.49/h**(Lambda)〜 **$2〜3/h**(RunPod/Vast on-demand)。AWS p5 は約$6.88/h、Azureは更に高い |
| B200 | $4.99〜5.89/h(Lambda / RunPod) |

→ $1000 は **A100 80GB で 500〜900時間相当、H100 で 330〜500時間相当**。3BモデルのLoRA実験としては十分な滑走路。

## 6. エンジニアリング上の注意

### 6.1 学習には80GB級が実質必須

- `CONTEXT = 24576`。`protocol.py:141` `chunk_ranges()` は
  `size = min((context - prefix_tokens - 3) // 2, CONTEXT)` でフレームを割る
  (codecとlatentが1フレーム2スロットを占めるため)
- 3分の曲 = 25Hzで約4,500 latentフレーム。AR+NARの**フルattention(O(n²))**で
  約9,000〜10,000トークン規模の系列を勾配付きで回す必要がある
- コミュニティ実装はすべて**10秒クリップ前提**(`clip_seconds` 既定10、VRAM 24GB)。
  full temporal coverageに広げる時点で**メモリ設計が別物になる**
- 必要な対策: gradient checkpointing、`song_chunks()` と同じチャンク境界での学習、
  潜在の事前キャッシュ(ai-toolkitも「潜在キャッシュ必須」と明記)
- **補足(後続調査)**: この節は「曲全体を1系列で回す」前提での見積り。NARはフローマッチング
  なので**20秒窓 + gradient checkpointing なら約12 GiB**で足り、80GB級は不要になる。
  具体的な数値は [lora-architecture-and-data-scale.md](lora-architecture-and-data-scale.md) §3.1〜3.3 を参照
- **補足2(環境の確定)**: 短窓設計なら **ローカルの RTX 4090(24GB)で学習・推論とも走る**。
  学習は S=1,000 × batch 16 で 11.72 GiB、推論は CFG 2分岐・context上限でも 11.92 GiB。
  [implementation-milestones.md](implementation-milestones.md) の実行環境節を参照
- **補足3(密maskは不要)**: `nar.py` の `CachedNAR` は密 attention mask を作らず、
  AR を causal で prefill して K/V をキャッシュし、NAR クエリを `cat(ar_k, nar_k)` に対して
  マスクなし全結合 attention で解くことで、ハイブリッドマスクと等価な結果を得ている。
  学習ループもこの構造を写すべき

### 6.2 再現性の作り込みを壊さないこと

- `nar.py:34` `song_chunks()` はノイズテンソルを**一度だけ**引いて各チャンクにビューを渡す
- `sampling.py` は「歴史的な算術を保存する」ためCFGの減算・乗算・加算をBF16で行う(`legacy_off` 分岐も同様)
- `SymbolicPlan.save()/load()` はSHA-256マニフェストで改変を検出する
- → 学習側で条件付け分布を推論側と食い違わせない。
  Voicesの `nar_train.py` が「NAR学習と推論が同じ条件付け分布を見るように、
  prefix cacheを読む前に歌手のARアダプタを適用する」としているのは、まさにこの原則

### 6.3 評価の作り込み

- PERは**ASR 4パスの最小値**(公式プロトコル)。単一転写でラベルを流用しない
- 話者類似度は **MERT cosineを鵜呑みにしない**。絶対値は同一性の百分率ではなく、
  同系統の曲なら別人でも0.85〜0.9に乗る。`pyannote/wespeaker-*` 系で測り直し、
  無伴奏ボーカルに分離 + 同一話者の別曲ペアで閾値校正を行う
- mel spectrogram loss は知覚品質と相関しない(Mothersuperiorの報告)。**LTAS距離**の方が信頼できた
- アブレーションの軸: delivery tag有無 / ジャンル一致 / 複数シード / アダプタ有無 / デコーダ選択
  (先行実装はここが単一ランで、0.003差がシードノイズに埋もれている)

## 7. 期待値の設定とゲート

- 到達目標を「**本人と聞き間違える**」ではなく
  「**声質の傾向と歌い回しが一致し、聴き手が同一シリーズと認識できる**」に置く
- 現行公開知見の最良値が stock 0.852〜0.888 / adapter 0.846〜0.885 である以上、
  **Phase 2(現 M3)が空振りする確率は実在する**
- **ゲート設計**: Phase 0.5 / Phase 1(現 **M2**)の結果を中断判断に使う。
  特に round-trip(対象話者の音源 → semantic token → NAR再生成)が
  聴感で破綻するなら、その上に載る学習はすべて土台を欠く
- **保険**: Phase 0(現 **M0**、表記実験・評価ハーネス)と Phase 3(現 **M6**、日本語 diction)は
  **話者同一性が達成できなくても独立して価値が残る**成果物。
- **現行の撤退条件**は各マイルストーンに定量条件付きで記載されている
  ([implementation-milestones.md](implementation-milestones.md))。
  日本語対応の改善は話者クローンより成功確率が高く、予算内で確実にリターンが出る投資

## 8. ライセンス(個人利用での結論)

| 対象 | ライセンス |
|---|---|
| 本リポジトリのコードと `skills/yue2-music` | **Apache 2.0**(`LICENSE`) |
| YuE2-3B / YuE2-Vae / YuE2-Vae-legacy の重み | **CC BY-NC 4.0**(`MODEL_LICENSE`) |
| 学習したLoRA/アダプタ | 重みの**派生物**なので CC BY-NC 4.0 の非商用条件に従う |

**個人利用では実質的な制約にならない。** `MODEL_LICENSE` の追加許諾(2026-09-16)より:

> 個人ユーザー、コンテンツクリエイター、個人の資格で活動するミュージシャンに対し、
> モデル重みを無償で使って出力を生成し、それらの出力を**公開・配布・販売・ライセンス・その他の方法で
> 収益化する**許可を与える。これらの個人的な創作活動について、licensorはNonCommercial制限を
> 必要な範囲で放棄する。別途の商用ライセンスもライセンス料もロイヤリティも不要。

ただし:

- この許諾は**出力の生成と利用**に関するもので、**重みの商用再配布・販売には及ばない**
- **企業による重みの商用利用は対象外**(別途 ryuanab@connect.ust.hk と契約)
- **責任ある利用が許諾の条件**:
  > 違法・有害・欺瞞的・非倫理的な目的(詐欺、ハラスメント、**deceptive impersonation** を含む)に
  > 使ってはならない。入力・出力・利用について、**必要な権利と同意の取得を含め**利用者が責任を負う
- → 対象話者が本人以外なら、**本人の同意(実演家の権利・パブリシティ権)の確認が前提**。
  学習素材の権利処理も同様
- 生成物は「YuE2で生成したという理由だけでNonCommercial制限の対象にはならない」
- クレジット(`#YuE2` タグやモデル名)は**推奨だが出力に対しては条件ではない**。
  ただし**重みや適応済み重みを共有する場合の帰属要件は放棄されない**

## 9. 参考リンク

**公式**
- [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) / [YuE2 v0.1.6 release](https://github.com/multimodal-art-projection/YuE/releases/tag/yue2-v0.1.6)
- [デモ: map-yue2.github.io](https://map-yue2.github.io/) / [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) / [WildSongBench](https://huggingface.co/datasets/m-a-p/WildSongBench)
- [Issue #141(LoRA学習の前提が公開されていない)](https://github.com/multimodal-art-projection/YuE/issues/141) / [Issue #60(Finetuning the model)](https://github.com/multimodal-art-projection/YuE/issues/60)
- YuE v1 論文: [arXiv:2503.08638](https://arxiv.org/html/2503.08638v1)(dual-track music ICLの記述)

**コミュニティ**
- [Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)
- [Mothersuperior/YuE2-hum-to-song](https://huggingface.co/Mothersuperior/YuE2-hum-to-song)
- [KadoBOT/ComfyUI-YuE2-Voices](https://github.com/KadoBOT/ComfyUI-YuE2-Voices) / [rambrogi/YuE2-Voices-training-kit](https://huggingface.co/rambrogi/YuE2-Voices-training-kit)
- [Starnodes2024/ComfyUI-YuE2-Trainer](https://github.com/Starnodes2024/ComfyUI-YuE2-Trainer) / [Saganaki22 fork](https://github.com/Saganaki22/ComfyUI-YuE2-Trainer)
- [ostris/ai-toolkit PR #1042](https://github.com/ostris/ai-toolkit/pull/1042) / [RunComfy: YuE2 LoRA Training](https://www.runcomfy.com/trainer/ai-toolkit/yue2-lora-training)
- [ComfyUI Wiki: YuE2 Real-Audio Encoder](https://comfyui-wiki.com/en/news/2026-09-16-yue2-realaudio-encoder)
- [github.com/topics/yue2](https://github.com/topics/yue2)

**日本語**
- [GIGAZINE: 音楽生成AI「YuE2」が無料公開される](https://gigazine.net/gsc_news/en/20260911-yue2-music-generation-ai/)
- [YouTube: 音楽生成AI「YuE2」の日本語楽曲の例](https://www.youtube.com/watch?v=mijyEq2v2Fs)
