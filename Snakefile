include: "workflow/00_dataprep.smk",
include: "workflow/10_get_embeds.smk"

rule all:
	input:
		"data/combined_df.csv",
		"data/embeds_df.csv"
