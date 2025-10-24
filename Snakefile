include: "workflow/00_dataprep.smk"

rule all:
	input:
		"data/combined_df.csv"
