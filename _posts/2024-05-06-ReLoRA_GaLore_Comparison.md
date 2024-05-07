## Parameter Efficient Pre-Training (PEPT)

Large language models (LLMs) have revolutionized natural language processing (NLP) tasks, but their massive parameter count creates challenges: high computational cost and limited accessibility. Parameter-efficient fine-tuning (PEFT) methods have addressed this by reducing the resources needed to fine-tune LLMs for specific tasks. This raises the question: can we achieve similar efficiency gains during the pre-training stage?

Parameter-efficient pre-training (PEPT) is an emerging area of research that explores techniques for pre-training LLMs with fewer parameters. PEPT has the potential to significantly reduce the computational cost associated with pre-training, which is typically very resource-intensive signifying the importance of the research in this area. ReLoRA [1] is the first parameter-efficient training method to pre-train large language models. ReLoRA uses LoRA decomposition, merges and resets the values of the LoRA matrices multiple times during training, increasing the total rank of the update. Another recent advance in PEPT is GaLore [2]. In GaLore, the gradient is projected into its lower rank form, updated using an optimizer, and finally projected back to its original size, reducing the memory requirement for pre-training by a huge margin.

This blog discusses ReLoRA and GaLore and the similarities and differences between them.

## ReLoRA
coming soon ...

## GaLore
coming soon ...

## Comparison of ReLoRA and GaLore
