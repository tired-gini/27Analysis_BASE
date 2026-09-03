# Improving Language Understanding by Generative Pre-Training (GPT-1) 논문 리뷰

**저자**: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (OpenAI, 2018)

---

## 1. 어떤 문제를 해결하려 했는가? (Abstract)

라벨 없는 텍스트(unlabeled text)는 인터넷에 넘쳐나지만, 특정 task(감정분석, 질의응답 등)를 학습시키려면 사람이 직접 라벨을 붙인 데이터가 필요한데 이건 항상 부족하다. 그래서 이 논문은 **"라벨 없는 대량의 텍스트로 먼저 언어모델을 사전학습(generative pre-training)하고, 그 다음 각 task에 맞게 살짝만 손봐서 파인튜닝(discriminative fine-tuning)하자"**는 전략을 제안한다. 이 방식으로 12개 벤치마크 중 9개에서 기존 SOTA를 넘었다.

## 2. 연구 동기와 문제점은 무엇인가? (Introduction)

- 딥러닝은 라벨 데이터가 많이 필요한데, 분야마다 라벨 데이터가 부족한 경우가 많음.
- 기존에도 word2vec, GloVe 같은 "단어 임베딩"으로 비지도 학습의 이점을 봤지만, 이건 단어 단위 정보만 활용한 거라 한계가 있음.
- 단어보다 더 큰 단위(문장, 문단)의 정보를 라벨 없는 텍스트에서 뽑아 쓰는 건 두 가지 이유로 어려웠음:
  1. **어떤 학습 목표(objective)가 전이(transfer)에 가장 효과적인지 불명확** — language modeling, 번역, discourse coherence 등 방법마다 task별로 결과가 달랐음.
  2. **학습한 표현을 target task로 옮기는 가장 좋은 방법에 대한 합의가 없음** — task마다 구조를 새로 짜야 하거나 복잡한 학습 스킴이 필요했음.

즉, "비지도 사전학습을 어떻게 해야 여러 task에 두루 잘 통하는 범용 표현을 얻을 수 있는가"가 이 논문의 출발점.

## 3. 관련 연구 동향은 어떠한가? (Related Works)

- **준지도 학습(semi-supervised learning)**: 단어/구 단위 통계를 feature로 쓰는 초기 접근 → word embedding 활용 → 이 논문은 문장/문서 수준의 더 상위 의미(semantics)를 잡으려 함.
- **비지도 사전학습**: 이미지 분류, 음성인식 등에서 먼저 시도됨. 언어모델링 목표로 사전학습 후 파인튜닝하는 방식은 Dai & Le, Howard & Ruder 등이 먼저 했지만 LSTM 기반이라 긴 문맥을 못 잡는 한계가 있었음. 이 논문은 **Transformer**를 써서 더 긴 range의 언어 구조를 잡는 게 차별점.
- **보조 학습 목표(auxiliary objective)**: language modeling을 보조 objective로 같이 쓰면 성능이 오른다는 선행연구(Rei 등)가 있었고, 이 논문도 이를 채택.

## 4. 연구 접근법과 모델 구조는 어떻게 되는가? (Method)

### 핵심 아이디어
학습을 **2단계**로 나눈다.

**1단계 – 비지도 사전학습(Unsupervised Pre-training)**
- 라벨 없는 대량 텍스트 코퍼스 U = {u₁, ..., uₙ}에 대해, 앞의 k개 토큰을 보고 다음 토큰을 맞추는 표준 언어모델링 목표를 최대화:

  L₁(U) = Σᵢ log P(uᵢ | uᵢ₋ₖ, ..., uᵢ₋₁; Θ)

- 모델은 **12-layer Transformer decoder** (masked self-attention, 768차원, attention head 12개). 원조 Transformer(Vaswani et al.)의 인코더-디코더 중 디코더만 떼어 쓴 형태라고 보면 됨 — RNN처럼 앞에서 뒤로만 보되, self-attention으로 먼 거리 단어도 직접 참조 가능.
- 비유하자면: 책을 엄청 많이 읽으면서 "다음에 무슨 단어가 나올까?"를 계속 맞춰보는 연습을 시키는 것. 이 과정에서 문법, 상식, 문맥 이해 능력이 자연스럽게 몸에 밴다.

**2단계 – 지도 파인튜닝(Supervised Fine-tuning)**
- 라벨 있는 데이터셋 C (입력 x¹...xᵐ, 정답 y)에 대해, 사전학습된 모델 위에 linear layer 하나(Wy)만 추가해서 y를 예측:

  P(y | x¹,...,xᵐ) = softmax(hₗᵐ · Wy)

- 여기에 language modeling을 **보조 목표**로 같이 넣으면 일반화도 잘 되고 수렴도 빨라짐 (auxiliary LM):

  L₃(C) = L₂(C) + λ · L₁(C)   (λ=0.5)

- 즉 파인튜닝 때 추가되는 파라미터는 Wy와 구분자(delimiter) 토큰 임베딩뿐 — 구조를 거의 안 바꾼다는 게 핵심 장점.

### Task별 입력 변환 (Task-specific Input Transformations)
서로 다른 구조를 가진 task 입력을, 하나의 연속된 토큰 시퀀스로 바꿔서 모델 구조 변경 없이 처리 (traversal-style approach):

| Task | 변환 방식 |
|---|---|
| 텍스트 분류 | [Start, Text, Extract] 그대로 |
| 자연어 추론(entailment) | [Start, 전제(premise), 구분자($), 가설(hypothesis), Extract] |
| 유사도(similarity) | 순서가 없으므로 두 문장 순서를 바꿔서 두 번 넣고 결과를 더함 |
| 질의응답/상식추론 | [문서, 질문, $, 후보답변]을 답변 개수만큼 각각 독립적으로 넣고 softmax로 정규화 |

## 5. 실험 구성과 결과는? (Experiments)

**사전학습 데이터**: BooksCorpus (미출간 책 7,000권 이상, 장르 다양) — 문장 단위로 섞이지 않고 긴 연속된 문맥이 살아있는 게 핵심 (경쟁 데이터셋인 1B Word Benchmark는 문장 단위로 섞여있어서 장거리 문맥 학습에 불리).

**결과 요약**:
- **자연어추론(NLI)**: MNLI, SciTail, QNLI, SNLI 4개 데이터셋에서 SOTA 갱신 (최대 +5.8%p). RTE는 데이터가 작아서 이전 멀티태스크 모델보다 낮음(56.0% vs 61.7%).
- **질의응답/상식추론**: Story Cloze +8.9%p, RACE +5.7%p — 특히 긴 문맥을 다루는 데 강점.
- **의미 유사도**: STS-B +1점, QQP +4.2%p 개선.
- **분류(Classification)**: CoLA(문법성 판단) 45.4점으로 이전 최고(35.0)를 크게 앞지름. SST-2는 91.3%로 SOTA급.
- **GLUE 종합 점수**: 72.8 (이전 최고 68.9 대비 개선).
- 총 12개 중 **9개 task에서 SOTA 달성**.

## 6. 무엇을 발견했고 한계점은 무엇인가? (Discussion / Analysis)

- **레이어 전이 실험**: 사전학습된 레이어를 많이 옮겨줄수록(transfer) 성능이 계속 좋아짐 → 각 레이어가 나름의 유용한 기능을 담고 있다는 증거.
- **Zero-shot 성능**: 파인튜닝 없이도 사전학습만으로 여러 task를 어느 정도 풀 수 있고, 사전학습이 진행될수록 이 zero-shot 성능도 꾸준히 오름 → 언어모델링 자체가 여러 언어이해 능력을 부산물로 학습한다는 근거.
- **Ablation(제거 실험)**:
  - 보조 LM 목표 제거 시: 큰 데이터셋(NLI, QQP)에서는 성능 하락, 작은 데이터셋에서는 오히려 소폭 상승 → 데이터가 클 때만 보조 목표가 도움됨.
  - Transformer 대신 LSTM 사용 시: 평균 5.6점 하락 → Transformer의 attention 구조가 전이에 유리.
  - 사전학습 자체를 제거(처음부터 지도학습만): 평균 14.8%p 큰 폭 하락 → 사전학습이 성능의 핵심 원천.
- **한계점**: RTE처럼 작은 데이터셋에서는 멀티태스크 학습 모델에 밀림. 논문에서도 "멀티태스크 학습을 추가하면 더 좋아질 것"이라 언급하지만 이 논문에서는 시도하지 않았다고 명시.

## 7. 결론 및 주요 요약은? (Conclusion)

하나의 task-agnostic 모델(사전학습된 Transformer)을 다양한 task에 최소한의 구조 변경만으로 적용해도, task별로 따로 설계한 모델들을 능가할 수 있음을 보였다. 핵심은 "긴 문맥을 가진 텍스트로 사전학습 + Transformer 구조"의 조합이며, 이는 이후 GPT-2, GPT-3, GPT-4 시리즈로 이어지는 "사전학습-파인튜닝(→ 나중엔 프롬프팅)" 패러다임의 출발점이 되었다.

## 8. 기존 연구 대비 차별점은 무엇인가?

- 이전 연구(Peters et al., ELMo 등)는 사전학습된 표현을 **보조 feature**로 붙이고 task별 구조를 새로 설계 → 파라미터가 task마다 새로 많이 필요.
- 이 논문은 사전학습된 모델 자체를 파인튜닝하며, task별로 **입력만 다르게 변환**하고 모델 구조는 거의 그대로 둠 → 훨씬 적은 task-specific 파라미터.
- 사전학습 objective로 language modeling만 단순하게 사용했고, RNN/LSTM이 아니라 **Transformer decoder**를 처음으로 이 방식에 적용.

## 9. 핵심 아이디어는?

> "라벨 없는 텍스트로 언어의 일반적인 패턴을 먼저 익히고(사전학습), 그 지식을 각 task에 살짝만 맞춰서(파인튜닝) 재사용하자."

비유하면, 다양한 책을 많이 읽어서 언어 감각과 상식을 키운 사람에게, 특정 시험(NLI든 QA든) 문제 형식만 몇 문제 보여주면 금방 적응해서 잘 푸는 것과 같다. "형식 적응"에 필요한 데이터는 적어도 되고, "언어 감각"은 이미 갖춰져 있기 때문.

## 10. 주요 수식과 코드 분석

**(1) 언어모델링 목표**
```
L1(U) = Σ log P(u_i | u_{i-k}, ..., u_{i-1}; Θ)
```
- 직전 k개 토큰을 보고 다음 토큰의 확률을 최대화. GPT 계열의 "다음 단어 맞추기" 학습의 근본.

**(2) Transformer 순전파**
```
h0 = U·We + Wp          # 토큰 임베딩 + 위치 임베딩
h_l = transformer_block(h_{l-1})   # l = 1...n
P(u) = softmax(h_n · We^T)          # 출력층은 입력 임베딩과 가중치 공유(weight tying)
```
- 마지막 출력층 We^T가 입력 임베딩 We를 그대로 재사용하는 weight tying 기법 사용 (파라미터 절약 + 성능 도움).

**(3) 파인튜닝 목표**
```
P(y|x1,...,xm) = softmax(h_l^m · Wy)
L2(C) = Σ log P(y|x1,...,xm)
L3(C) = L2(C) + λ·L1(C)     # λ=0.5, 보조 LM 목표
```
- h_l^m: 마지막 토큰(m번째)의 마지막 레이어 hidden state → 이걸로 문장 전체를 대표시켜 분류.

**하이퍼파라미터 요약**
- 12-layer Transformer decoder, hidden 768, head 12, feedforward 3072
- Adam, max lr 2.5e-4, warmup 2000 step, cosine decay
- BPE vocab 40,000, dropout 0.1, GELU 활성함수
- 배치 64, 시퀀스 길이 512, 100 epoch (사전학습)
- 파인튜닝: lr 6.25e-5, 배치 32, 3 epoch 정도로 충분

## 11. 기타 (공부 메모)

- 이 논문이 바로 **GPT-1**이며, 이후 GPT-2(더 큰 모델 + zero-shot 강조), GPT-3(few-shot in-context learning)로 스케일업 되는 시리즈의 시작점.
- BERT(2018, 같은 해 later)는 이 논문의 "단방향(왼→오)" 한계를 지적하며 양방향(masked LM)으로 대응해서 나온 논문이라, GPT-1과 비교해서 같이 읽으면 이해가 깊어짐.
- "Zero-shot Behaviors" 섹션은 이후 GPT-2/3에서 강조되는 in-context learning의 초기 씨앗이라고 볼 수 있음 — 이미 GPT-1에서 "파인튜닝 없이도 언어모델 자체가 task를 어느 정도 푼다"는 관찰이 나왔다는 점이 흥미로움.
