# 日本語対応の現状と独自にやる価値

調査日 2026-09-18。

## 1. 結論

| 問い | 回答 |
|---|---|
| YuE2は日本語で歌えるか | **歌える**。公式デモに日本語ボーカル曲が複数ある |
| トークナイザの拡張は必要か | **不要**。Qwen tiktoken(151,643 ordinary tokens)で日本語はネイティブに扱える |
| 日本語の品質は測られているか | **測られていない**。公式ベンチに言語別内訳がない |
| phoneme/発音を明示制御できるか | **できない**。`phonemes` フィールドが存在しない |
| 独自にやる価値はあるか | **ある。最も成功確率が高い領域**。話者同一性が達成できなくても独立して価値が残る |

## 2. テキスト側の前提 — Qwen BPEをそのまま凍結して使用

`src/yue2/tokenization_yue2.py`(41行)の実装:

- クラスのdocstring: 「The frozen text/ABC BPE. **This is not the audio tokenizer.**」
- `qwen.tiktoken` の ordinary token 数が **151,643** であることを厳格に検証(不一致なら
  `ValueError("Expected checkpoint-native qwen.tiktoken (151643 ordinary tokens)")`)
- 特殊トークン: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<R>`, `<S>`, `<X>`, `<mask>`, `<sep>`
  + `<extra_0..199>`、うち index 204,205 を `<abc>`, `</abc>` に差し替え
- `encode()` は **unicodedata.NFC正規化**を適用してから `encode_ordinary()`
- 正規表現パターンは `\p{L}` / `\p{N}` ベースなので、CJKも通常の文字クラスとして処理される

→ Qwenの多言語BPEなので**日本語(漢字・かな)はそのままトークン化される**。
語彙拡張やトークナイザ差し替えは不要であり、むしろ `151643` の厳格検証があるため**やってはいけない**。

`vocab_size = 184704` の内訳(`src/yue2/protocol.py:7-12`):

```
        0 .. 151642  テキストBPE(Qwen ordinary tokens)
   151643            EOD (<|endoftext|>)
   151847, 151848    ABC_START, ABC_END
   151851, 151852    MUSIC_START, MUSIC_END
   151853 .. 184620  codec tokens (CODEC_OFFSET + 32,768)
   184621 .. 184623  LATENT_START, LATENT_END, LATENT_PAD
```

## 3. 公式の日本語対応状況

**わかっていること**

- 公式デモサイト([map-yue2.github.io](https://map-yue2.github.io/))に**日本語ボーカル曲のサンプルが複数掲載**されている
- [GIGAZINE(2026-09-11)](https://gigazine.net/gsc_news/en/20260911-yue2-music-generation-ai/) が
  「楽譜から楽曲生成可能＆**日本語ボーカルも対応**」として報道。ただし**定量値は記事に含まれていない**
- `docs/generation.md:7` の指示:「genre, instruments, vocal character, **language**, and tempo を `style` に入れる」
  → 言語は自然文で `style` に書く。`examples/song.json` も `"style": "English, warm piano pop, ..."` と
  **言語名を先頭に置く書式**になっている
- 多言語の混在も可能とされる

**わかっていないこと**

- `docs/benchmarks.md` の WildSongBench は **192プロンプト・17設定**で、YuE2のPERは **8.44%**(best-of-8は9.79%)。
  **言語別内訳は非公開**。日本語プロンプトが何件含まれるかも公開されていない
- 外部のハンズオン報告では言語ごとに品質差があるとされる(ロシア語・ベンガル語・ブラジルポルトガル語・
  フランス語・インドネシア語・タガログ語は良好、ヒンディー語・ウルドゥー語は明確に弱い)が、
  **日本語の定量評価は見つからなかった**

→ **日本語PERは誰も測っていない。最初に測った人が最初に改善できる。**

## 4. 構造的な制約 — 発音を明示制御する口がない

- `skills/yue2-music/references/generation-and-covers.md:17`
  「**`phonemes`** のリクエストフィールドは存在しない」
  「発音や音符アラインメントの指示は検証用サイドカーに留めること。**それらはハード条件付け入力ではない**」
- したがって日本語の発音は**基盤モデルの暗黙知識に依存**する。
  Qwen BPEは漢字をそのままトークン化するため、**漢字の読み(音読み/訓読み、固有名詞)は本質的に曖昧**。

**→ 最初にやるべき無料実験**: 同一歌詞を

1. 通常表記(漢字かな混じり)
2. **全ひらがな**
3. **全カタカナ**
4. **ローマ字**

の4通りで生成し、PERを比較する。学習ゼロ・コストゼロで、効果があれば**運用ルールとして即座に使える**。
`skills/yue2-music/references/abc-editing.md:126` が推奨する「section / lyric line / word・syllable /
phonemes / stress / 選択voice / note index / pitch / 予定onset・durationを記録するサイドカー」を
日本語モーラ単位で作れば、そのまま検証データになる。

## 5. 日本語歌唱に固有の技術課題

### 5.1 歌詞アライメントのパイプラインが英語前提

コミュニティの学習パイプライン(`cursor_prep.py`)は **Demucs + MMS** で歌詞をアラインし、
`lyric-cursor weight 0.08` で条件付けしている(§ [community-implementations](community-implementations.md) §1.1)。

日本語では次が素直に通らない:

- **語境界がない**(分かち書きしない)ため word-level アライメントの前提が崩れる
- **漢字→読みの変換が必須**。既存研究でも「漢字とかなの混在テキストをかな列に変換する際の誤りを
  ユーザーが手動修正できるようにする」という設計が採られている
- **モーラ単位の扱い**: 促音(っ)、撥音(ん)、長音(ー)、二重母音、母音の無声化は
  「1音素=1音符」にならず、音符への割り当て規則が英語の syllable/stress とは別物
- 歌唱は音素長とリズム構造の変動が大きく、話し声向けアライナ(MFA等)の精度が落ちる

→ **日本語向けにアライメント段を作り直すことは、明確に切り出せる独自実装項目**。
`pyopenjtalk` 等でかな・モーラ列に落とし、モーラ単位でカーソル条件を作るのが素直な設計。

### 5.2 ABC側の注意

`skills/yue2-music/references/abc-editing.md:124`(言語変更時のガイダンス)がそのまま該当する:

> 言語を変えるときは、逐語訳ではなく**歌えるように歌詞を作り直す**。
> 最終スコアの確定したvocal noteを使って、歌われる全音節を意図した音符またはメリスマに対応させる。
> 子音をアタック/リリースに、母音をメリスマにわたって持続させる。**1音素は1音符ではない**。
> 語強勢、短い音符での子音密度、長い音符での母音選択、ブレスの休符、ピックアップ音節、セクションタグを確認する。

ネイティブABCダイアレクトは `Vocal` / `Ins` の2声で、`w:` 歌詞フィールドは**サポート外**
(`abc-editing.md:80` が明示的に拒否する対象に `lyric w: fields` が含まれる)。
→ 歌詞と音符の対応はABCに書けないので、`lyrics` の文字列とABCの音符列の整合は**外部で検証するしかない**。

## 6. 評価ハーネスは手元資産でほぼ組める

調査時点でローカルのHFキャッシュに存在する資産:

| 用途 | 使えるモデル |
|---|---|
| 日本語ASR(PER測定。公式プロトコルは**4パス取って最小値**) | `kotoba-whisper-v2.0-mlx`、`litagin/anime-whisper`、`mlx-community/parakeet-tdt_ctc-0.6b-ja`、`Qwen3-ASR-1.7B`、`Systran/faster-whisper-large-v3`、`whisper-large-v3-turbo` |
| 話者類似度(MERT cosineより妥当な指標) | `pyannote/wespeaker-voxceleb-resnet34-LM`、`pyannote/speaker-diarization-community-1` |
| 日本語歌唱データ | `jaCappella`(日本語アカペラ・コーラス、**最初からボーカルのみ**) |
| 音声表現 | `ntu-spml/distilhubert`、`Aratako/Semantic-DACVAE-Japanese-32dim` |

PER測定の注意(`skills/yue2-music/references/listening-and-evaluation.md:94`):

> checked PERプロトコルは**ASR 4パス**を実行する。各パスの転写、選択結果、1パス目の結果を保存し、
> スコアを報告する際にプロトコルを明示すること。**単一の転写に置き換えてラベルだけ流用してはならない**。
> 参照言語と実際の新歌詞を確認し、スカラー値を完全な検証として扱わず残存誤りを検査すること。

話者類似度の注意: 歌唱は話者照合モデルの想定外条件なので、
**①無伴奏ボーカルに分離してから測る ②同一話者の別曲ペアで閾値を校正する**の2つを前提にする。

## 7. 独自実装として切り出せる項目

| # | 項目 | コスト | 独立した価値 |
|---|---|---|---|
| 1 | 日本語4表記のPER比較(学習不要) | **$0** | ◎ 運用ルールとして即使える |
| 2 | 日本語PER評価ハーネス(ASR 4パス + 話者類似度) | **$0**(ローカル) | ◎ 以降の全実験の土台 |
| 3 | 日本語モーラ単位の歌詞アライメント段 | 実装工数のみ | ◎ 学習パイプラインの前提 |
| 4 | 日本語 diction AR LoRA(full temporal coverage実装で) | $0〜200 | ◎ 話者同一性と独立 |
| 5 | tokenizer head の日本語音源への適応(`joint.py` 相当) | $0〜80 | ○ 話者学習の土台 |

コストは**ローカルの RTX 4090(24GB)で回せる分を $0 として更新済み**
([implementation-milestones.md](implementation-milestones.md) の実行環境節)。

→ 実行計画への組み込みは [implementation-milestones.md](implementation-milestones.md) の **M6**
(項目1〜2は **M0** に含まれる)。旧 [custom-training-plan.md](custom-training-plan.md) §5 の
Phase 3 が M6 に相当する。
