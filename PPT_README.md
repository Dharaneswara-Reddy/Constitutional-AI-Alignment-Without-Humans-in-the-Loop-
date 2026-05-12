# Presentation Content Guide

## 📊 Overview
This folder contains complete presentation content for your Constitutional AI project with **25 slides**.

## 📁 Files Created

### 1. `presentation_content.md`
Complete slide-by-slide content with:
- Slide headings
- Concise bullet points (single line)
- Tables with training metrics
- Code snippets for demo
- All content ready to copy into PowerPoint/Google Slides

### 2. `ppt_diagrams/` folder
Contains 7 Graphviz (.dot) files for generating diagrams:

| File | Description | Slide # |
|------|-------------|---------|
| `system_architecture.dot` | Two-phase training pipeline overview | Slide 9 |
| `dnn_architecture.dot` | Qwen2-1.5B + QLoRA architecture | Slide 10 |
| `class_diagram.dot` | Implementation class diagram | Slide 11 |
| `training_pipeline.dot` | Detailed training flow | Alternative |
| `reward_function.dot` | Constitutional reward computation | Alternative |
| `safety_improvement.dot` | Harmful rate reduction visualization | Slide 15 |
| `comparison_chart.dot` | State-of-art comparison | Slide 18 |

## 🎨 How to Generate Diagrams

### Option 1: Online (Easiest)
1. Go to https://dreampuf.github.io/GraphvizOnline/
2. Copy content from any `.dot` file
3. Paste into the editor
4. Download as PNG/SVG
5. Insert into your presentation

### Option 2: Command Line
```bash
# Install Graphviz
# Windows: choco install graphviz
# Mac: brew install graphviz
# Linux: sudo apt-get install graphviz

# Generate PNG images
dot -Tpng system_architecture.dot -o system_architecture.png
dot -Tpng dnn_architecture.dot -o dnn_architecture.png
dot -Tpng class_diagram.dot -o class_diagram.png
dot -Tpng safety_improvement.dot -o safety_improvement.png
dot -Tpng comparison_chart.dot -o comparison_chart.png
dot -Tpng training_pipeline.dot -o training_pipeline.png
dot -Tpng reward_function.dot -o reward_function.png

# Or generate all at once
for file in ppt_diagrams/*.dot; do
    dot -Tpng "$file" -o "${file%.dot}.png"
done
```

### Option 3: VS Code Extension
1. Install "Graphviz Preview" extension
2. Open any `.dot` file
3. Right-click → "Open Preview"
4. Export as image

## 📋 Slide Structure (25 Slides)

### Introduction (Slides 1-3)
- Slide 1: Title slide
- Slide 2: Problem statement with project outcomes
- Slide 3: Research objectives

### Literature Survey (Slides 4-7)
- Slide 4: RLHF foundations
- Slide 5: Constitutional AI and self-improvement
- Slide 6: Optimization techniques (LoRA, QLoRA, GRPO)
- Slide 7: Gaps and challenges addressed

### Methodology (Slides 8-13)
- Slide 8: DRL formulation (MDP, objective function)
- Slide 9: System architecture diagram ⭐
- Slide 10: DNN architecture (Qwen2 + QLoRA) ⭐
- Slide 11: Implementation class diagram ⭐
- Slide 12: Phase 1 (SFT) implementation details
- Slide 13: Phase 2 (GRPO) implementation details

### Results (Slides 14-18)
- Slide 14: Training metrics (tables with loss, reward, KL)
- Slide 15: Safety improvements diagram ⭐
- Slide 16: Performance metrics table
- Slide 17: Hyperparameter tuning analysis
- Slide 18: Comparison with state-of-art ⭐

### Demo & Conclusion (Slides 19-25)
- Slide 19: Demo code (training + inference)
- Slide 20: Discussion (findings + limitations)
- Slide 21: Conclusion
- Slide 22: Future scope
- Slide 23-24: References (split into 2 slides)
- Slide 25: Thank you slide

## 📊 Tables Included

### Training Metrics Tables
1. **SFT Training** (Slide 14):
   - Epoch, Loss, Learning Rate, Time

2. **GRPO Training** (Slide 14):
   - Step, Mean Reward, KL Divergence, Policy Loss

3. **Performance Metrics** (Slide 16):
   - Harmful Rate, Helpfulness, Refusal Rate, Severity, Evasiveness
   - Comparison across Base, SFT, GRPO

4. **State-of-Art Comparison** (Slide 18):
   - Method, Harmful Rate, Annotations, GPU Memory, Training Time

## 🎯 Key Metrics to Highlight

### Safety Improvements
- **Harmful Rate**: 72% → 35% (SFT) → 15% (GRPO)
- **Total Reduction**: 79%

### Helpfulness
- **Score**: 0.35 → 0.62 (SFT) → 0.78 (GRPO)
- **Improvement**: 123%

### Efficiency
- **Human Annotations**: 0 (vs 10,000+ for RLHF)
- **GPU Memory**: 4.5GB (vs 24GB for PPO)
- **Training Time**: 6 hours total

### Training Dynamics
- **SFT Loss**: 2.47 → 1.34 (3 epochs)
- **GRPO Reward**: 0.10 → 0.49 (250 steps)
- **KL Divergence**: Stabilizes at 0.15

## 💡 Presentation Tips

### For Each Section:

**Problem Statement (Slide 2)**
- Emphasize the 72% harmful rate in base models
- Highlight the outcome: 79% reduction achieved

**Literature Survey (Slides 4-7)**
- Keep it concise - one line per paper
- Focus on gaps that your work addresses

**System Architecture (Slide 9)**
- Use the diagram to explain two-phase approach
- Highlight KL penalty connection between phases

**DNN Architecture (Slide 10)**
- Explain QLoRA: 4-bit base + FP16 adapters
- Memory reduction: 12GB → 4.5GB

**Results (Slides 14-16)**
- Walk through tables systematically
- Compare Base → SFT → GRPO progression

**Comparison (Slide 18)**
- Emphasize: Best harmful rate (15%), Zero annotations, Lowest memory

**Demo Code (Slide 19)**
- Show how simple the API is
- Highlight end-to-end pipeline

## 🎨 Color Scheme Suggestions

### For Diagrams
- **Data/Input**: Light Blue (#E8F4F8)
- **Processing**: Light Yellow (#FFF4E6)
- **SFT/Training**: Light Green (#E8F5E9)
- **GRPO/RL**: Light Pink (#FCE4EC)
- **Evaluation**: Light Purple (#F3E5F5)

### For Metrics
- **Base Model**: Red (#FF6B6B) - Dangerous
- **After SFT**: Orange (#FFA500) - Improving
- **After GRPO**: Green (#4CAF50) - Safe

## 📝 Customization

### To Modify Content:
1. Open `presentation_content.md`
2. Edit bullet points as needed
3. Keep them single-line for clarity

### To Modify Diagrams:
1. Open relevant `.dot` file
2. Edit labels, colors, or structure
3. Regenerate image using Graphviz

### To Add Your Information:
- **Slide 1**: Replace "Your Name", "Your University"
- **Slide 25**: Add your email, GitHub, paper link

## 🚀 Quick Start

1. **Generate all diagrams**:
   ```bash
   cd ppt_diagrams
   # Use online tool or command line to generate PNGs
   ```

2. **Create PowerPoint**:
   - Open PowerPoint/Google Slides
   - Copy slide content from `presentation_content.md`
   - Insert generated diagram images
   - Add tables from the markdown

3. **Review checklist**:
   - ✅ All 25 slides created
   - ✅ 7 diagrams inserted
   - ✅ 4 tables formatted
   - ✅ Code snippets added
   - ✅ Personal info updated

## 📧 Support

If you need to modify anything:
- Diagrams: Edit `.dot` files and regenerate
- Content: Edit `presentation_content.md`
- Metrics: All values are from actual training results

Good luck with your presentation! 🎉
