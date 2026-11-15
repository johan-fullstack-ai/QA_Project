# QA Rule Pack

## QA principles

These rules describe broader QA principles and is a framework for evaluating model output generation.

---

### Source Usage

Source usage is judged based on whether the model's answer stays within the information found in the sources provided. A response should be treated as failing this principle when it presents information that is not found in any of the sources, refers to a source that does not exist, or claims that multiple sources support something that is only found in one of them. The judgment focuses on whether the answer is grounded in the source text, not on whether the source is current or complete.

---

### Consistency analysis

Consistency analysis assesses whether the model responds in a way that is stable and predictable across multiple runs. A model that gives different answers to the same question does not have to be wrong, but the variation must not mean that it contradicts itself or changes position without the source text justifying it. Consistency is therefore assessed based on whether the answers are within the same semantic framework, if they are based on the same parts of the source material, and if they do not introduce new facts that are missing from the sources. Variation in wording is allowed, as is variation in level of detail, but variation in facts or conclusions is a sign of instability. A response should be treated as failing this principle when it gives conflicting facts across runs, switches between different interpretations of the source text without the source text supporting it, or shows a pattern of adding or removing central parts of the information. The assessment focuses on whether the model follows a stable line, not on whether it expresses itself identically.

---

### Structure and Format

Structure and format are assessed based on whether the model presents its answer in a way that is clear, coherent, and appropriate to the question. The model does not have to follow a specific template, but it must not provide an answer that is disorganized, fragmented, or lacking logical order. An answer should have a natural progression where information comes in a reasonable sequence, and where the model does not jump between topics or mix in irrelevant parts. The format should be consistent within the answer, and the model should avoid mixing styles or providing incomplete lists, broken sentences, or abrupt interruptions. A response should be treated as failing this principle when its structure makes the content difficult to understand, it lacks context, or its format is so unclear that it affects interpretation. The assessment focuses on whether the answer is readable and logical, not on whether it follows a specific style guide.

---

### Risk Rating

Risk rating is used to assess how likely it is that an answer contains inaccuracies, hallucinations, or lacks anchoring in the source material. A low-risk answer is one where the model stays close to the source text, where facts are clearly anchored, and where wording is stable across runs. A medium-risk answer is one where the model varies the answers too much between runs. A high-risk answer is one where the model presents facts that are missing from the sources, where it refers to sources that do not exist, or where it shows a pattern of introducing new information that cannot be verified. Note that the risk rating does not determine whether an answer is correct or incorrect, but rather indicates the level of risk of the answer.

---

### Overall QA Principle
This QA framework describes how responses from a language model should be assessed when used in prototypes and internal experiments. The aim is to create a method that makes it possible to evaluate quality, stability and anchoring in source material without being dependent on specific tools or implementations. The framework is therefore formulated as a set of principles and classifications that can be used regardless of the model, data source or workflow. The assessment focuses on whether the model adheres to the information it has been given, whether it responds consistently across multiple runs, whether the structure of the response makes it understandable and useful, and whether there are signs that the model introduces information that cannot be verified. The result of the QA assessment should be used to determine whether a prompt needs to be adjusted, whether the model needs to be controlled differently or whether the workflow needs to be changed to give the model better conditions. The framework is intended to be reusable in future prototypes and should serve as a stable basis for understanding how a model behaves, why it does so and how to improve its response in a controlled manner.

---

### Failure Modes
Failure modes describe the typical ways in which a language model can produce answers that deviate from the source material or from the expected logic of the workflow. The assessment is not about finding individual mistakes, but about understanding patterns that show when the model tends to go wrong. A common failure mode is when the model fills in gaps with reasonable but unconfirmed assumptions when the source text is unclear or incomplete. Another is when the model mixes up different parts of the source material and presents them as if they belong together, even though they come from separate contexts. It also happens that the model overgeneralizes, where it takes a specific detail and turns it into a broad rule, or that it presents conclusions that sound correct but are not supported by the source. Failure modes are therefore used to determine whether the prompt needs to be made clearer, whether the model needs to be controlled more tightly, or whether the workflow needs to provide the model with more structured information before it can provide a reliable answer.