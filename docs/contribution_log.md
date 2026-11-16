# Individual contribution log

**Charith Leburu (SE23UARI074) – project lead, RQ1, pipeline and experiments**
I led the project and ran the full pipeline on the university DGX cluster for Qwen2.5-7B and Llama-3.1-8B, including data generation, dataset freezing and experiments. I owned RQ1: hidden-state extraction, linear probes, baselines, controls, ablations, layer-wise analysis and minimal-pair comparisons, including documenting the uninformative control task. I prepared the optional RQ4 activation-steering and RQ5 ByteSized32 next-state uncertainty code, which was not run. I integrated notebook changes, fixed cluster issues, maintained GitHub, committed the pre-registration, logged deviations, generated figures, recorded the demo and drafted the report.

**Shanvi Rishi (SE23UARI136) – RQ2 analysis and quality annotation**
I worked through the RQ2 sections of the cluster notebook for Qwen2.5-7B and Llama-3.1-8B to understand how the stated-answer experiments were set up and run. I owned the RQ2 analysis, examining what both models say when asked directly about reversibility and comparing their answers with the linear-probe predictions using kappa. As annotator A, I independently labelled 100 QC pairs following the written instructions, without seeing the automatic labels. I also checked the RQ2 report section against the results and prepared the RQ2 presentation material.

**Rishikesh B (SE23UARI104) – RQ3 analysis and quality annotation**
I worked through the RQ3 sections of the cluster notebook for Qwen2.5-7B and Llama-3.1-8B to understand how the agent-behaviour experiments were set up and run. I owned the RQ3 analysis, examining agent episodes, caution measures, K, A and the blindness gap to assess how reversibility knowledge relates to actions. As annotator B, I independently labelled the same 100 QC pairs without seeing the automatic labels or annotator A's decisions. I also checked the RQ3 report section against the results and prepared the RQ3 presentation material.

**Srikar Karthik (SE23UCSE104) – sandbox and labeller review, references and presentation**
I studied the sandbox and reversibility-labelling code and the corresponding notebook sections, which produce the data used in the Qwen2.5-7B and Llama-3.1-8B experiments. I reviewed the sandbox settings, tools and minimal pairs, together with the reversibility labeller's undo search, exact versus approximate state keys and search horizon. I checked that the dataset and labelling documentation matched these components and prepared the corresponding presentation material. I also contributed to verifying bibliography entries against the original papers and reviewed the full report.

All four members read the final report and agree with its content.

**Use of AI tools:** Claude assisted with most of the initial code, documentation and report draft. The team checked, edited and integrated its output.
