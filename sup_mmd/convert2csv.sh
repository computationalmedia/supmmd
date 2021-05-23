dirr=$1
rm $dirr/gs*.csv
for f in $dirr/gs*.txt;
do
	echo "processing ${f}"
	touch "${f}.csv"
	# mmdpq_tac08-B_x.x_g1.0_b0.02_a0_SFbxc_r0.001_ep181 ROUGE-1 Average_F: 0.34015 (95%-conf.int. 0.32361 - 0.35762)
	echo "cv,gamma,beta,alpha,lambdaa,r,epoch,eval,metric,score" >> $f.csv
	cat $f | perl -ane 'print "$5,$6,$7,$8,$9,${10},${11},${12},${14},${15}\n" if m#^mmdpq?-?(comp[01]\.lin[12])?_(tac08|tac09|duc03|duc04)-([AB])_([xc])\.(\d+|x)_g(\d\.?\d*)_b(\d\.?\d*)_a(\d+)_?l?(\d\.?\d*)?_SF[a-z]{3}_r(\d\.?\d*)_ep(\d+)\s(ROUGE-(1|2|SU4))\sAverage_([R]):\s(\d\.\d+).+#' >> $f.csv
done
