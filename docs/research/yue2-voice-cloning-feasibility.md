# YuE2本体の実装調査 — 参照音声とボイスクローンの可否

調査対象: 本リポジトリ `src/yue2/`(`yue2-infer`)の実コード。調査日 2026-09-18。

## 1. 公式リポジトリは推論専用

`src/yue2/*.py` は全13ファイル・約3,800行。学習ループ・PEFT/LoRA・アダプタ実装は**一切存在しない**。

| ファイル | 行数 | 役割 |
|---|---:|---|
| `modeling_yue2.py` | 705 | MoTバックボーン、AR forward、`nar_velocity()` |
| `modeling_vae.py` | 589 | VAE(48kHzステレオ潜在のデコード) |
| `fast.py` | 426 | vLLMバックエンド |
| `pipeline.py` | 398 | `plan()` / `generate_semantic()` / `synthesize()` / `decode()` |
| `progress.py` | 316 | 進捗表示 |
| `nar.py` | 261 | NARのチャンク化とODE solve |
| `cuda_graph.py` | 240 | ARのCUDA Graph |
| `sampling.py` | 154 | トークンサンプリングループ |
| `protocol.py` | 145 | `SongRequest`、プロンプト契約、特殊トークン |
| `quantization.py` / `storage.py` / `tokenization_yue2.py` / `__init__.py` | 114 / 148 / 41 / 15 | 量子化、重み解決、テキストBPE |

`lora|peft|finetune|adapter|speaker|timbre|reference_audio` を全文検索しても、ヒットするのは
推論時のガード(`cuda_graph.py:44`、`nar.py:241` の `if model.training: raise`)と
HuggingFaceの `adapter_kwargs` 除外(`modeling_vae.py:420`)だけである。

## 2. リクエスト契約 — 話者を指定する口が存在しない

`src/yue2/protocol.py:81` `SongRequest` が受けるフィールド:

```python
style: str          # ジャンル、楽器、vocal character、言語、テンポを文章で
lyrics: str         # [Verse] / [Chorus] 等のセクションタグ付き歌詞
cot: str = "full"   # "full" | "melody" | "off"(記号的プランの粒度)
seed: int = 831001
abc: str | None     # 外部ABCスコアを直接与える(cot=full/melody 必須)
cfg_scale: float | None
id: str = "song"
```

`skills/yue2-music/references/generation-and-covers.md:17` に明記されている通り:

> There is no request field for `reference_audio`, `phonemes`, `bpm`, `negative_prompt`,
> an edit interval or a **reference singer**.

プロンプトは `protocol.py:108` `SongRequest.text()` で次の1本のテキストに組まれる:

```
{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n
```

→ **話者埋め込みも参照音声もアーキテクチャに存在しない**。声はこの `style` 文と基盤モデルのpriorだけで決まる。

## 3. アーキテクチャ — 音色はNAR分岐にしか入らない

### 3.1 Mixture-of-Transformers によるAR/NARの重み分離

`src/yue2/modeling_yue2.py:227` `DecoderLayer` は "full MoT" であり、**28層すべてでAR経路とNAR経路が別パラメータ**:

```python
self.input_layernorm / self.self_attn          # AR attention path
self.nar_input_layernorm / self.nar_self_attn  # NAR attention path (別Q/K/V/O)
self.post_attention_layernorm / self.mlp       # AR MLP path
self.nar_pre_mlp_layernorm / self.nar_mlp      # NAR MLP path
```

`forward()` は `ar_mask` でトークン位置ごとにルーティングする。
→ LoRAのターゲットとして `self_attn.*`/`mlp.*`(AR=作曲・意味)と
`nar_self_attn.*`/`nar_mlp.*`(NAR=音響・音色)を**明確に撃ち分けられる**。

主要コンフィグ(`modeling_yue2.py:61` `YuE2Config` 既定値):
`hidden_size=2048`、`num_hidden_layers=28`、`num_attention_heads=16`、`num_key_value_heads=8`、
`head_dim=128`、`intermediate_size=6144`、`vocab_size=184704`、`max_position_embeddings=24576`、
`latent_type="vae"`、`latent_dim=64`、`max_latent_frames=24576`。

### 3.2 NARの条件付け経路

`src/yue2/modeling_yue2.py:609` `nar_velocity()` が flow matching の速度場 v_θ(x_t, t) を計算する。要点:

- 潜在は `vae2llm`(64→2048の線形)で hidden に射影され、**timestep embedding と latent position embedding を加算**して
  **全NAR位置**(`LATENT_START` / content / `LATENT_END`)に注入される(`:665` のコメント「matching training's torch.where」)。
- ハイブリッドattention mask: `AR→AR` は causal、**`NAR→AR` は全参照**、`NAR→NAR` は双方向、`AR→NAR` は遮断(`:670-683`)。
  → **NARはARの全hidden statesを見ている**。トークンIDだけでなくAR側の重み変化も音響出力に伝わる。
- 出力は `llm2vae` で 2048→64 に戻す。
- `nar_cond_end > 0` のとき **text-only モード(codec dropout)**: NARは `position < nar_cond_end` のテキスト部分と
  NAR位置のみを見る(`:676-679`)。`nar.py:121` が `visible_length` として実装。
  → **音声→semantic tokenエンコーダが無くてもNAR側を条件付けして学習できる経路が、推論コードに既に存在する**。
  コミュニティのNARトレーナーはこの regime を使っている。

### 3.3 semantic token は音色を運べない(帯域の計算)

`src/yue2/protocol.py:11`:

```python
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
```

tokenizer head の出力レートは **25 Hz**(コミュニティ実装のモデルカード記載、MERT-v2-FullSong layer-20 特徴と同レート)。

> log2(32768) = 15 bit / frame × 25 frame/s = **375 bps**

375bpsは「音声内容」レベルの帯域であり、**声の個性を運べる情報量ではない**。
`sampling.py:33` の通り semantic フェーズのロジットは `[CODEC_OFFSET, CODEC_OFFSET+CODEC_SIZE)` + `MUSIC_END` に制限されるため、
この隘路は設計上のものである。

**帰結**: 出力の音色は **NAR(flow matching)が AR hidden states と自身のpriorから生成**している。
音色を制御したいなら、①NARに音色を運ぶ入力経路を作るか、②NARの重みに焼き込むかのどちらかしかない。
AR側にLoRAを当てる現行のコミュニティ手法は、375bpsの隘路の手前で戦っている。

## 4. 参照音声を渡せるか — 3つのレベル

### レベル1: 公式API → **不可能**

- `SongRequest` に該当フィールドが無い(§2)。
- `pipeline.py:271` `generate_semantic()` は `token_prefixes(request, tokenizer, plan.abc_ids) != plan.prefix` で
  `ValueError("Plan prefix disagrees with request/exact ABC IDs")` を投げる。
- `pipeline.py:285` `synthesize()` も同様に
  `ValueError("Semantic result does not retain the request's exact prefix")` を投げる。
  → **公式パイプラインは任意トークンの差し込みを意図的に塞いでいる**。
- `SymbolicPlan.load()`(`pipeline.py:44`)はSHA-256マニフェストで保存済みプランの改変も検出する。
- さらに **音声→semantic tokenエンコーダがそもそも未リリース**。公式配布物だけでは参照音声を入力形式に変換できない。

### レベル2: 公式のcover経路 → **メロディだけ運べる。声は運べない**

`docs/covers.md` の手順は audio →(SheetSage2で転写)→ ABC → `abc=` で再生成。

- `skills/yue2-music/SKILL.md:85`「melody condition であり、**元歌手のidentityや波形は保持しない**」
- `docs/editing.md:51`「スコアが変わらなくても、編集範囲外の歌唱・音色・波形の同一性は保証しない」
- `docs/benchmarks.md` のカバー評価(SHS100K 948作品、3,792出力/手法)も「一般の生成チェックポイントを使い、
  原曲-カバーのペア教師もカバー専用のfine-tuningも与えていない」と明記。

### レベル3: 低レベル関数を直接叩く → **ICL注入は書ける(約20行)**

公式パイプラインの下層は開いている:

- `src/yue2/sampling.py:57` `generate_tokens(model, prefix, sampling, seed, phase, ...)` は**任意のprefixリスト**を受ける。
  検証は `len(prefix) + sampling.max_tokens > CONTEXT`(=24576)のみ。
- `src/yue2/nar.py:228` `synthesize(model, prefix, codec, seed, ...)` も任意の prefix/codec を受ける。
  `song_chunks()`(`nar.py:34`)の検証は範囲チェック(`max(codec) < CODEC_SIZE` 等)のみ。

したがって次の構成が可能:

```python
ref_tokens = tokenizer_head(target_singer_audio)          # コミュニティ tokenizer head v9 が必要
prefix = plan.prefix + [t + CODEC_OFFSET for t in ref_tokens]
new_tokens, *_ = generate_tokens(model, prefix, sampling, seed, "semantic")
latents = nar_synthesize(model, prefix, ref_tokens + new_tokens, seed)
audio = pipe.decode(latents)
```

**前提**: 音声→semantic tokenエンコーダが必須 = コミュニティの tokenizer head(v9)。
**素の公式モデル単体では、この経路すら成立しない。**

### レベル3の見込み — 声が乗る確率は低い

- §3.3 の帯域制約(375bps)がそのまま効く。
- `YuE2-hum-to-song` の著者も「**文脈のsemantic tokenが既にメロディを固定しているので、carrierはタイミングとピッチのニュアンスしか足せない**」
  と述べ、声は意図的にサイン波で捨てている([community-implementations.md](community-implementations.md) §1.4)。
- ただし **YuE v1 には dual-track ICL があった**(分離したvocal/accompanimentをトークンレベルでインターリーブし、
  20〜40秒のセグメントをランダムサンプリングして前置して**学習**していた)。公式も当時「ICLがLoRAに最も近い」と案内していた。
  YuE2ではAPIから消えており、**YuE2が codec-prefix ICL で学習されたかは非公開**。
  → ここが唯一の未知であり、$0〜50で決着する実験になる(Phase 0.5)。

## 5. 公式に欠けているピースと公式の立場

| 欠けているもの | 状況 |
|---|---|
| 音声 → semantic token エンコーダ | 未リリース。YuE v1 時代から同じ指摘([Issue #141](https://github.com/multimodal-art-projection/YuE/issues/141)) |
| 学習コード | 未リリース([Issue #60](https://github.com/multimodal-art-projection/YuE/issues/60)) |
| 参照音声/話者条件付け | アーキテクチャに存在しない |

公式の回答は「**ICLがLoRAに最も近い代替であり、学習を必要としない**」というもので、YuE v1 の文脈で示された。
YuE2ではそのICL自体がAPIから外れている。

## 6. テキスト側トークナイザ(日本語の前提条件)

`src/yue2/tokenization_yue2.py` は **Qwen の tiktoken BPE をそのまま凍結して使用**:

- `qwen.tiktoken` の ordinary token 数が **151,643** であることを厳格に検証(不一致なら例外)
- 特殊トークン: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<R>`, `<S>`, `<X>`, `<mask>`, `<sep>` + `<extra_0..199>`、
  うち index 204,205 を `<abc>`, `</abc>` に差し替え
- `encode()` は **NFC正規化**を適用してから `encode_ordinary()`
- クラスのdocstring: 「This is not the audio tokenizer」

→ 日本語はこの多言語BPEでネイティブにトークン化される。**トークナイザの拡張は不要**。
詳細は [japanese-support.md](japanese-support.md)。

## 7. ステージAPIとチャンク境界(学習設計に影響)

`pipeline.py` の4段構成: `plan()` → `generate_semantic()` → `synthesize()` → `decode()`。

- `protocol.py:141` `chunk_ranges(frames, prefix_tokens, context)`:
  `size = min((context - prefix_tokens - 3) // 2, CONTEXT)`
  → codec と latent が1フレームあたり2スロットを占めるため、コンテキスト24,576では
  1チャンクあたり約12,000フレーム弱 = 25Hzで約8分。通常の曲は1チャンクに収まる。
- `nar.py:34` `song_chunks()` はノイズテンソルを**一度だけ**引いて各チャンクにビューを渡す(再現性のため)。
- `decode()` は既定で `decode_tiled(core_frames=..., halo_frames=16)` のタイル復号。

学習をフル曲・full temporal coverage で行う場合、この境界に合わせた設計と gradient checkpointing が必要になる。
→ [custom-training-plan.md](custom-training-plan.md) のエンジニアリング注意を参照。
