The folder contains 7 files:
1. codereview.jsonl: CodeReview dataset, as well as the results of testing the CodeReviewer model and ChatGPT.
2. codereview_new.jsonl: CodeReview New dataset, as well as the results of testing the CodeReviewer model and ChatGPT.
3. testset_500cases.jsonl: Experimental results for RQ1.
4. trainset_1000cases.jsonl: Experimental results for RQ1 conducted on the CodeReview trainset.
5. RQ3_RQ4_score.jsonl: Experimental results for RQ3 and RQ4.
6. RQ3_codereviewer_rootcause.jsonl: Root cause analysis for CodeReviewer errors in RQ3.
7. The final_website folder contains the code for executing RQ1 and RQ2. RQ3 and RQ4 involve manual labeling of certain categories and subsequent statistical analysis, so there is no code involved.

The field explanations for codereview and codereview_new are as follows:
	old: the code snippet to be modified
	review: the review suggestion
	new: the modified code snippet
	commit_url: the URL of the submission
	gpt_answer: the complete answer provided by ChatGPT
	gpt_code: the code part in ChatGPT's answer
	model_code: the code prediction given by the CodeReviewer model
	language: the programming language of the code
	gpt_em: whether gpt_code exactly matches new (EM)
	gpt_em_trim: whether the trimmed gpt_code exactly matches new (EM-Trim)
The field explanations for the RQ1 cases are as follows:
	new_answer_p_a_t: the answer provided by ChatGPT with the p-th prompt, the a-th attempt, and the temperature of t (20 for 2.0, 15 for 1.5, 10 for 1.0, 5 for 0.5, 0 for 0), for example, new_answer_1_2_5 represents the answer with the first prompt, the second attempt, and the temperature of 0.5
	new_code_p_v_t: the code part in the answer provided by ChatGPT with the pvt combination as described above.
It should be noted that due to the significant decline in the quality of generated results when using high temperatures, this experiment did not repeat the runs 10 times for temperature=1.5 and 2.