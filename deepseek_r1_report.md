# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning 논문 리뷰

**저자**: DeepSeek-AI (2025, arXiv:2501.12948)

---

## 1. 어떤 문제를 해결하려 했는가? (Abstract)

지금까지 LLM의 추론(reasoning) 능력을 끌어올리려면 대개 사람이 직접 만든 chain-of-thought(단계별 풀이) 데이터로 학습시켜야 했다. 이 논문은 **사람이 만든 추론 데이터 없이, 순수 강화학습(RL)만으로도 LLM의 추론 능력을 이끌어낼 수 있다**는 걸 보여준다. 그렇게 학습된 모델은 자기반성(self-reflection), 검증(verification), 전략을 바꿔가며 푸는 능력을 스스로 익혔고, 수학·코딩·STEM 같은 검증 가능한 task에서 기존 지도학습 기반 모델을 능가했다. 또한 이 큰 모델이 익힌 추론 패턴을 작은 모델에도 옮겨(distillation) 성능을 끌어올릴 수 있음을 보였다.

## 2. 연구 동기와 문제점은 무엇인가? (Introduction)

- LLM이 충분히 커지면 추론 능력이 "emergent(창발적으로)" 나타난다는 건 알려져 있었지만, 이걸 사전학습만으로 얻으려면 막대한 연산이 필요함.
- Chain-of-thought(CoT) 프롬프팅으로 보완할 수 있지만, 결국 **사람이 만든 예시나 라벨링된 추론 궤적에 의존**한다는 근본적 한계가 있음.
- 사람이 만든 예시에 의존하면, 모델의 성능이 결국 "사람이 생각하는 방식"에 갇혀버려서, 사람보다 더 나은(non-human-like) 추론 경로를 스스로 탐색하지 못함.
- 그래서 저자들은 "사람 라벨링을 최소화하고, RL을 통한 자기진화(self-evolution)로 추론 능력을 키울 수 있을까?"를 탐구함.

핵심 접근: DeepSeek-V3-Base 위에 **GRPO(Group Relative Policy Optimization)**라는 RL 알고리즘을 적용하고, 보상(reward)은 오직 "최종 답이 맞았는가"만 본다. 추론 과정 자체에는 아무 제약을 두지 않음. 이렇게 만든 모델이 **DeepSeek-R1-Zero**이며, SFT(지도 파인튜닝) 단계를 아예 건너뛴 게 특징. 이 모델은 답을 더 길게 생성하면서 검증·반성·대안 탐색 같은 행동을 자연스럽게 익혔다.

다만 R1-Zero는 가독성이 떨어지고 한 응답 안에서 영어·중국어가 섞이는 문제가 있어서, 이를 보완한 **DeepSeek-R1**(콜드스타트 데이터 + RL + SFT를 섞은 멀티스테이지 학습)을 추가로 개발함. 또한 이 추론 능력을 작은 모델로 증류(distill)해서 공개함.

## 3. 관련 연구 동향은 어떠한가?

이 버전(v2)의 논문 본문에는 별도의 "Related Work" 섹션이 따로 없고, 관련 연구 언급이 서론에 녹아들어 있음:
- **CoT 프롬프팅** (Wei et al., 2022b; Kojima et al., 2022): few-shot 예시나 "단계별로 생각해보자" 같은 간단한 프롬프트로 중간 추론 과정을 이끌어내는 기존 접근.
- **post-training 단계의 고품질 다단계 추론 학습** (Chung et al., 2024; OpenAI, 2023): 사람이 만든 궤적을 학습시키는 방식 — 이 논문이 극복하려는 대상.
- **PPO(Proximal Policy Optimization)** (Schulman et al., 2017): LLM RL 단계에서 널리 쓰이던 알고리즘이지만 리소스 소모가 큼 → 이를 단순화한 게 GRPO(Shao et al., 2024).

## 4. 연구 접근법과 모델 구조는 어떻게 되는가? (Method)

### 4.1 GRPO (Group Relative Policy Optimization)

PPO는 "가치함수(critic)"를 따로 학습시켜야 해서 연산 비용이 크다. GRPO는 이 critic을 없애고, **같은 질문에 대해 여러 개의 답을 한 그룹으로 샘플링**한 뒤, 그 그룹 안에서의 상대적 우수함으로 advantage를 계산한다.

```
A_i = (r_i - mean({r_1,...,r_G})) / std({r_1,...,r_G})
```

즉 "이 답이 같은 질문에 대한 다른 답들보다 평균적으로 얼마나 잘했는가"를 표준화한 값을 보상 신호로 쓰는 것. 비유하면, 같은 시험문제를 여러 번 풀어보게 하고, 그 중 상대적으로 잘 푼 답안에 가산점을 주는 방식 — 별도의 채점위원(critic 모델)을 안 키워도 됨.

전체 목적함수(PPO의 clipped objective + KL penalty와 유사한 구조):

```
J_GRPO(θ) = E[ (1/G) Σ min( ratio·A_i, clip(ratio, 1-ε, 1+ε)·A_i ) - β·D_KL(π_θ || π_ref) ]
```

### 4.2 DeepSeek-R1-Zero — 순수 RL만으로

- 초기 모델: DeepSeek-V3-Base
- SFT 없이 바로 RL 적용
- 보상은 **규칙 기반(rule-based)** 두 가지만 사용:
  - **정확도 보상(accuracy reward)**: 수학은 정답 박스 형식으로 규칙 검증, 코딩은 컴파일러/테스트케이스로 검증
  - **형식 보상(format reward)**: `<think>...</think>` 태그 안에 추론 과정을 넣도록 유도
  - `Reward_rule = Reward_acc + Reward_format`
- **신경망 기반 보상 모델은 의도적으로 안 씀** — 대규모 RL에서 reward hacking(보상을 속이는 편법)에 취약하고, 재학습 비용도 크기 때문.
- 결과: AIME 2024 정확도가 학습 중 15.6% → 77.9%까지 상승. 학습이 진행될수록 응답 길이도 자연스럽게 늘어남 (더 오래 "생각"하게 됨).
- **"Aha moment"**: 모델이 스스로 풀이를 재검토하다가 "Wait, wait. That's an aha moment I can flag here"처럼 사람처럼 반성하는 표현을 자발적으로 쓰기 시작함. 이건 사람이 가르친 게 아니라 RL 과정에서 자연스럽게 나타난 행동.

### 4.3 DeepSeek-R1 — 멀티스테이지 파이프라인

R1-Zero의 가독성/언어혼용 문제를 해결하기 위해 4단계 파이프라인 구성 (Figure 2):

1. **콜드스타트(Cold Start)**: 사람이 정리한 소량의 대화형·읽기 좋은 CoT 데이터로 먼저 SFT (→ R1 Dev-1)
2. **1차 RL**: 추론 정확도/형식 보상 + **언어 일관성 보상**(목표 언어 단어 비율)을 추가해서 언어 혼용 문제 완화 (→ R1 Dev-2)
   ```
   Reward_language = Num(Words_target) / Num(Words)
   ```
3. **거절 샘플링 + SFT**: DeepSeek-V3와 사람이 함께 만든 데이터로 추론+비추론(글쓰기 등) 데이터를 섞어 다시 SFT (→ R1 Dev-3)
4. **2차 RL**: 다양한 프롬프트에 대해 규칙 기반 보상(추론) + **모델 기반 보상(선호도, helpfulness/harmlessness)** + 언어 일관성 보상을 합쳐서 최종 RL (→ DeepSeek-R1)
   ```
   Reward = Reward_reasoning + Reward_general + Reward_language
   Reward_reasoning = Reward_rule
   Reward_general = Reward_reward_model + Reward_format
   ```

### 4.4 모델 기반 보상 (Section 3.1)

- **Helpful Reward Model**: DeepSeek-V3로 응답 A/B 쌍을 생성하고, 4번씩 순서를 바꿔가며 평가해 위치 편향(positional bias)을 줄임. 점수 차이가 1을 넘는 쌍만 채택. 총 66,000쌍으로 학습.
- **Safety Reward Model**: 106,000개 프롬프트에 "안전/위험" 라벨을 붙여 point-wise 방식으로 학습.

## 5. 실험 구성과 결과는? (Experiment)

**평가 벤치마크**: MMLU 계열, IF-Eval, GPQA Diamond, SimpleQA, SWE-Bench, LiveCodeBench, Codeforces, AIME 2024, CNMO 2024, C-Eval 등 영어/코드/수학/중국어 전반.

**단계별 성능 비교 (Table 3, 발췌)**:

| 벤치마크 | R1-Zero | R1-Dev1 | R1-Dev2 | R1-Dev3 | R1(최종) |
|---|---|---|---|---|---|
| AIME 2024 (Pass@1) | 77.9 | 59.0 | 74.0 | 78.1 | **79.8** |
| MATH-500 (Pass@1) | 95.9 | 94.2 | 95.9 | 95.4 | **97.3** |
| Codeforces (Rating) | 1444 | 1534 | 1687 | 1746 | **2029** |
| AlpacaEval2.0 (LC-winrate) | 24.7 | 50.1 | 55.8 | 62.1 | **87.6** |
| ArenaHard | 53.6 | 77.0 | 73.2 | 75.6 | **92.3** |

**해석**:
- R1-Zero → Dev1(콜드스타트 SFT 추가): instruction-following은 확 좋아지지만, 콜드스타트 데이터가 적어서 AIME 같은 순수 추론 성능은 오히려 일시적으로 떨어짐.
- Dev1 → Dev2(1차 RL): 코딩·수학·STEM 등 추론이 필요한 벤치마크에서 크게 개선. 반면 AlpacaEval 같은 범용 선호도 벤치마크는 소폭 개선에 그침 → **추론 지향 RL은 추론 능력만 집중적으로 끌어올린다**는 걸 시사.
- Dev2 → Dev3(비추론 데이터 포함 SFT): 일반 글쓰기/코드 엔지니어링 능력 개선.
- Dev3 → 최종 R1(2차 RL): AlpacaEval 25%p, ArenaHard 17%p 개선 — 주로 사용자 선호도/instruction-following 쪽에서 큰 향상.

## 6. 무엇을 발견했고 한계점은 무엇인가? (Discussion / Conclusion)

**발견**:
- 사전학습된 체크포인트는 이미 복잡한 추론을 할 잠재력을 갖고 있고, 이를 끌어내는 열쇠는 대규모 사람 라벨링이 아니라 **어려운 문제 + 신뢰할 수 있는 검증기(verifier) + 충분한 연산**이라는 것.
- 자기검증, 반성 같은 정교한 행동이 명시적으로 가르치지 않아도 RL 과정에서 유기적으로 나타남.

**한계점(논문이 스스로 밝힌 것)**:
- **구조화 출력/도구 사용**: 검색엔진, 계산기 같은 툴을 아직 못 씀.
- **토큰 효율성**: 쉬운 문제에도 필요 이상으로 길게 생각하는 "overthinking" 현상.
- **언어 혼용**: 중국어·영어 외 다른 언어 질의에서는 여전히 영어로 추론해버리는 경향.
- **프롬프트 민감성**: few-shot 프롬프팅이 오히려 성능을 떨어뜨림 → zero-shot으로 직접 문제를 설명하는 걸 권장.
- **소프트웨어 엔지니어링**: 평가 시간이 길어서 대규모 RL을 충분히 적용하지 못해 V3 대비 큰 개선이 없음.
- **Reward Hacking**: 순수 RL은 보상 신호의 신뢰성에 의존하는데, 글쓰기처럼 규칙 기반 보상을 만들기 어려운 task는 모델 기반 보상이 필요하고, 이건 학습이 진행될수록 "보상을 속이는 편법"에 취약해짐.

## 7. 결론 및 주요 요약은?

DeepSeek-R1-Zero는 **사람이 만든 추론 데이터 없이 순수 RL만으로도** LLM이 정교한 추론 행동(반성, 검증, 대안 탐색)을 스스로 익힐 수 있음을 증명한 첫 사례. 다만 가독성·언어혼용 문제가 있어서, 콜드스타트 SFT + 다단계 RL을 결합한 DeepSeek-R1으로 이를 보완했고, 최종적으로 추론 성능과 일반 선호도(helpfulness) 양쪽 모두를 잡았다. 이 추론 능력은 작은 모델로도 증류(distillation)해서 옮길 수 있다.

## 8. 기존 연구 대비 차별점은 무엇인가?

- 기존 연구들은 CoT 능력을 얻기 위해 **사람이 만든 추론 궤적으로 SFT**를 거치는 게 당연한 전제였음. 이 논문은 그 전제를 깨고 **SFT 없이 RL만으로** 추론 능력을 끌어냈다는 게 가장 큰 차별점(R1-Zero).
- 보상 모델(neural reward model) 대신 **규칙 기반 보상**만으로 대규모 RL을 안정적으로 돌렸다는 점 — reward hacking 문제를 원천적으로 피함.
- PPO 대신 **GRPO**를 써서 별도의 critic 없이 RL 비용을 줄임.

## 9. 핵심 아이디어는?

> "모델에게 정답을 맞히는 법을 직접 가르치지 말고, 정답 여부만 알려주는 보상을 주고 충분히 시행착오를 겪게 하면, 모델 스스로 더 나은 추론 전략(반성, 검증, 재시도)을 찾아낸다."

비유하면, 학생에게 풀이 과정을 일일이 알려주는 대신 "정답인지 아닌지"만 채점해주고 문제를 계속 풀게 시키면, 학생이 스스로 "어, 이거 아닌 것 같은데 다시 봐야겠다"는 습관을 자연스럽게 터득하는 것과 비슷하다. 사람이 짜준 풀이 방식에 얽매이지 않으니, 오히려 사람보다 더 나은 방식을 찾아낼 여지가 생긴다.

## 10. 주요 수식과 코드 분석

**(1) GRPO 목적함수**
```
J_GRPO(θ) = E[q~P(Q), {o_i}~π_θold(O|q)]
  (1/G) Σ min( ratio_i·A_i, clip(ratio_i, 1-ε, 1+ε)·A_i ) - β·D_KL(π_θ || π_ref)

ratio_i = π_θ(o_i|q) / π_θold(o_i|q)
```
- PPO의 clipped surrogate objective와 거의 같은 구조지만, **critic(가치함수) 없이** 그룹 내 상대 점수로 advantage를 대체한 게 핵심.

**(2) Advantage 계산 (critic 없이)**
```
A_i = (r_i - mean({r_1,...,r_G})) / std({r_1,...,r_G})
```
- 같은 질문에 대한 G개의 답 중, 이 답이 평균보다 얼마나 잘했는지를 표준화. 별도 가치함수 학습이 필요 없어서 연산이 가벼움.

**(3) KL penalty**
```
D_KL(π_θ || π_ref) = π_ref(o_i|q)/π_θ(o_i|q) - log(π_ref(o_i|q)/π_θ(o_i|q)) - 1
```
- 정책이 참조 모델(π_ref)에서 너무 멀리 벗어나지 않도록 잡아주는 항.

**(4) 규칙 기반 보상**
```
Reward_rule = Reward_acc + Reward_format
```
- 정답 여부(acc) + `<think>` 태그 형식 준수(format) 두 개만 단순하게 더함. 복잡한 보상 설계 없이도 추론 능력이 창발했다는 게 이 논문의 핵심 메시지.

**(5) 언어 일관성 보상**
```
Reward_language = Num(Words_target) / Num(Words)
```
- CoT 안에서 목표 언어 단어 비율. 이걸 추가하면 정확도는 살짝 떨어지지만(ablation 결과) 사람이 읽기엔 더 자연스러워짐 — **성능과 가독성의 트레이드오프**를 명시적으로 다룬 부분.

**(6) 최종 보상 결합**
```
Reward = Reward_reasoning + Reward_general + Reward_language
Reward_reasoning = Reward_rule
Reward_general   = Reward_reward_model + Reward_format
```

**학습 하이퍼파라미터 (R1-Zero 기준)**
- lr 3e-6, KL 계수 0.001, temperature 1
- 질문당 16개 샘플, max length 32,768 → 8.2k step 이후 65,536으로 확장
- 배치 512 (32 질문 × 16 샘플), 총 10,400 step (≈1.6 epoch)
- 400 step마다 reference model을 최신 policy로 교체

## 11. 기타 (공부 메모)

- GPT-1 리포트에서 다뤘던 "사전학습 → 파인튜닝" 패러다임과 비교하면 흥미로운 계보가 보임: GPT-1/BERT는 **사전학습(비지도) + 지도 파인튜닝**이었다면, DeepSeek-R1은 여기서 한 단계 더 나아가 **사전학습 + 지도학습 없는 순수 RL**로도 능력을 끌어낼 수 있음을 보여줌. "라벨 데이터 의존을 줄인다"는 큰 흐름이 이어지는 셈.
- GRPO는 PPO의 경량화 버전이라, RL 알고리즘 자체를 공부할 때 PPO(Schulman et al., 2017)를 먼저 이해하고 GRPO의 차이점(critic 제거, 그룹 상대평가)을 보면 이해가 빠름.
- "Aha moment" 부분은 논문에서도 꽤 강조하는 대목이라, RL로 학습된 모델이 사람이 명시적으로 가르치지 않은 행동을 스스로 획득한다는 예시로 발표/토론 자료에 인용하기 좋은 부분.
- 이 리뷰는 pdftotext로 텍스트 레이어만 추출해서 정리한 것으로, Table 3의 그래프(Figure 1, 9)나 원문 서식은 시각적으로 확인하지 않았음. 그래프 자체(정확도-스텝, 응답길이-스텝 곡선)를 봐야 하는 발표라면 원문 PDF의 Figure 1을 직접 캡처해서 쓰는 걸 추천.
