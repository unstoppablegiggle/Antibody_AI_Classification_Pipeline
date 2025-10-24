rule prepdata:
	input:
		df1 = "data/SRR8283601_1_Heavy_Bulk.csv",
		df2 = "data/TheraSAbDab_SeqStruc_onlineDownload.csv"
	output:
		out = "data/combined_df.csv"
	shell:
		"python src/data_prep.py --df1 {input.df1} --df2 {input.df2} --out {output.out}" 
