# Diagram to Slide Mapping Guide

## 📊 Visual Guide: Which Diagram Goes Where

---

## Slide 9: System Architecture
**File**: `ppt_diagrams/system_architecture.dot`

**Purpose**: Show the complete two-phase training pipeline

**What it shows**:
```
HH-RLHF Dataset → Critique+Revision (Groq) → Phase 1: SFT (π_ref) → Phase 2: GRPO (π_θ) → Evaluation
                                                    ↓                           ↑
                                                    └─────── β·KL penalty ──────┘
```

**When to use**: 
- Explaining overall system flow
- Showing how SFT and GRPO connect
- Highlighting KL penalty mechanism

**Talking points**:
- "We start with 160K samples from HH-RLHF"
- "Use Groq API to generate critiques and revisions"
- "Phase 1 creates reference policy π_ref"
- "Phase 2 optimizes π_θ with KL penalty to prevent drift"

---

## Slide 10: DNN Architecture
**File**: `ppt_diagrams/dnn_architecture.dot`

**Purpose**: Show the neural network structure with QLoRA

**What it shows**:
```
Input Tokens
    ↓
Embedding Layer (151,936 vocab)
    ↓
28 Transformer Layers ←─── LoRA Adapters (r=16, 9.4M params)
    ↑                       
4-bit Quantization
    ↓
Output Logits
```

**When to use**:
- Explaining model architecture
- Showing how QLoRA works
- Demonstrating memory efficiency

**Talking points**:
- "Qwen2-1.5B base model with 28 transformer layers"
- "4-bit quantization reduces memory from 12GB to 4.5GB"
- "LoRA adapters add only 9.4M trainable parameters (0.6%)"
- "This enables training on consumer GPUs"

---

## Slide 11: Implementation Class Diagram
**File**: `ppt_diagrams/class_diagram.dot`

**Purpose**: Show software architecture and component relationships

**What it shows**:
```
Config ←── SFTTrainer ←── GRPOTrainer ──→ ConstitutionalReward
  ↑            ↓              ↓                      ↓
  └── DataPreparation    CheckpointManager      Evaluator
```

**When to use**:
- Explaining code organization
- Showing modular design
- Demonstrating implementation structure

**Talking points**:
- "Modular design with 7 main components"
- "Config centralizes all hyperparameters"
- "SFTTrainer creates reference policy"
- "GRPOTrainer uses ConstitutionalReward for scoring"
- "CheckpointManager enables training resumption"

---

## Slide 15: Safety Improvement
**File**: `ppt_diagrams/safety_improvement.dot`

**Purpose**: Visualize harmful rate reduction across training phases

**What it shows**:
```
Base Model (72% Harmful) ──51% reduction──→ After SFT (35% Harmful) ──57% reduction──→ After GRPO (15% Harmful)
                         └────────────────── 79% total reduction ──────────────────────┘
```

**When to use**:
- Presenting main results
- Showing training progression
- Highlighting achievement

**Talking points**:
- "Base model: 72% harmful - unacceptable for deployment"
- "After SFT: 35% harmful - 51% improvement"
- "After GRPO: 15% harmful - 57% further improvement"
- "Total: 79% reduction in harmful responses"

---

## Slide 18: Comparison with State-of-Art
**File**: `ppt_diagrams/comparison_chart.dot`

**Purpose**: Compare our method with RLHF and DPO

**What it shows**:
```
Base Model (72% harmful)
    ├── RLHF (PPO): 18-25% harmful, 10K annotations, 24GB, 48h
    ├── DPO: 20-28% harmful, 5K annotations, 16GB, 24h
    └── CAI (Ours): 15% harmful, 0 annotations, 4.5GB, 6h ✓ BEST
```

**When to use**:
- Comparing with baselines
- Showing competitive advantage
- Demonstrating efficiency

**Talking points**:
- "RLHF achieves 18-25% harmful but needs 10,000 annotations"
- "DPO improves efficiency but still needs 5,000 annotations"
- "Our method: 15% harmful with ZERO annotations"
- "We also use 5x less memory and train 8x faster"

---

## Alternative/Optional Diagrams

### Training Pipeline (Alternative to System Architecture)
**File**: `ppt_diagrams/training_pipeline.dot`

**Purpose**: More detailed view of training flow with subprocesses

**Use if**: You want to show more implementation details than Slide 9

**What it shows**:
- Data preparation steps (download, sample, critique, revise)
- SFT training loop (load, add LoRA, train, save)
- GRPO training loop (init, generate, score, update)
- Evaluation step

---

### Reward Function (Additional Technical Detail)
**File**: `ppt_diagrams/reward_function.dot`

**Purpose**: Explain constitutional reward computation in detail

**Use if**: Audience asks "How exactly does the reward work?"

**What it shows**:
```
Prompt + Response
    ↓
Sample 4 Constitutional Principles
    ↓
Parallel Evaluation (4 judges)
    ↓
Score Each (SAFE=1.0, PARTIAL=0.5, HARMFUL=0.0)
    ↓
Mean Score
    ↓
Soft Clamp [0.4, 0.6]
    ↓
Final Reward
```

---

## 🎨 Diagram Generation Instructions

### For Each Diagram:

1. **Open the .dot file** in a text editor
2. **Copy all content**
3. **Go to**: https://dreampuf.github.io/GraphvizOnline/
4. **Paste** the content
5. **Download** as PNG (recommended) or SVG
6. **Insert** into PowerPoint slide

### Recommended Settings:
- **Format**: PNG (better compatibility)
- **Resolution**: 300 DPI (high quality)
- **Size**: Let PowerPoint resize (maintain aspect ratio)

### PowerPoint Insertion:
1. Insert → Picture → From File
2. Resize to fill 50-70% of slide
3. Center horizontally
4. Add caption below if needed

---

## 📐 Diagram Placement Tips

### Slide 9 (System Architecture):
- **Position**: Center of slide
- **Size**: 70% of slide width
- **Caption**: "Two-Phase Constitutional AI Training Pipeline"
- **Bullets**: Place to the right or below diagram

### Slide 10 (DNN Architecture):
- **Position**: Center-left of slide
- **Size**: 60% of slide width
- **Caption**: "Qwen2-1.5B with QLoRA Adapters"
- **Bullets**: Place to the right with key specs

### Slide 11 (Class Diagram):
- **Position**: Center of slide
- **Size**: 80% of slide width
- **Caption**: "Implementation Architecture"
- **Bullets**: Minimal - let diagram speak

### Slide 15 (Safety Improvement):
- **Position**: Top-center of slide
- **Size**: 60% of slide width
- **Caption**: "Harmful Response Rate Reduction"
- **Bullets**: Place below with helpfulness scores

### Slide 18 (Comparison):
- **Position**: Left side of slide
- **Size**: 50% of slide width
- **Table**: Place to the right with detailed metrics

---

## 🎯 Diagram Customization

### If Diagram is Too Complex:
1. Open the .dot file
2. Remove less important nodes
3. Simplify labels
4. Regenerate

### If Colors Don't Match Theme:
1. Edit `fillcolor` values in .dot file
2. Use your presentation color scheme
3. Regenerate

### If Text is Too Small:
1. Edit `fontsize` values in .dot file
2. Increase from 10 to 12 or 14
3. Regenerate

---

## 📊 Quick Reference Table

| Slide # | Diagram File | Purpose | Priority |
|---------|--------------|---------|----------|
| 9 | system_architecture.dot | Show pipeline | ⭐⭐⭐ Must Have |
| 10 | dnn_architecture.dot | Show model | ⭐⭐ Recommended |
| 11 | class_diagram.dot | Show code | ⭐⭐ Recommended |
| 15 | safety_improvement.dot | Show results | ⭐⭐⭐ Must Have |
| 18 | comparison_chart.dot | Show comparison | ⭐⭐⭐ Must Have |
| - | training_pipeline.dot | Alternative detail | ⭐ Optional |
| - | reward_function.dot | Technical detail | ⭐ Optional |

---

## ✅ Diagram Checklist

Before presentation:
- [ ] Generated at least 3 priority diagrams (slides 9, 15, 18)
- [ ] Inserted diagrams into PowerPoint
- [ ] Resized diagrams appropriately (50-70% of slide)
- [ ] Added captions below diagrams
- [ ] Tested visibility from back of room
- [ ] Practiced explaining each diagram
- [ ] Prepared pointer/laser for tracing flow

---

## 💡 Presentation Tips for Diagrams

### When Showing a Diagram:

1. **Introduce it**: "Here's our complete training pipeline..."
2. **Orient the audience**: "On the left we have... on the right..."
3. **Trace the flow**: Use pointer to follow arrows
4. **Highlight key parts**: "Notice the KL penalty connection here..."
5. **Connect to results**: "This pipeline achieved 79% reduction..."

### Example Script for Slide 9:

> "Here's our complete system architecture. [PAUSE]
> 
> We start with the HH-RLHF dataset - 160,000 samples of human conversations. [POINT]
> 
> We sample 2,000 and send them to Groq API to generate critiques and revisions. [POINT]
> 
> In Phase 1, we train with supervised learning to create our reference policy, pi-ref. [POINT]
> 
> In Phase 2, we use GRPO reinforcement learning to optimize pi-theta. [POINT]
> 
> Notice this dashed red line - that's the KL penalty that keeps our policy from drifting too far from the reference. This prevents reward hacking. [POINT]
> 
> Finally, we evaluate with an AI judge. [POINT]
> 
> This entire pipeline runs in 6 hours on a consumer GPU."

---

## 🚀 Final Tips

### Do:
✅ Use diagrams to explain complex concepts
✅ Trace flow with pointer/laser
✅ Pause after showing diagram
✅ Connect diagram to results
✅ Keep diagrams simple and clear

### Don't:
❌ Show diagram without explanation
❌ Read text from diagram verbatim
❌ Use overly complex diagrams
❌ Skip over important connections
❌ Assume audience understands immediately

---

Good luck! Your diagrams will make the presentation much clearer and more engaging! 🎨
