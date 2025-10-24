rule getembeds:
	input:
		df = "data/combined_df.csv"
	output:
		out = "data/embeds_df.csv"
	shell:
		"python src/get_embeds.py --df {input.df} --out {output.out}"
