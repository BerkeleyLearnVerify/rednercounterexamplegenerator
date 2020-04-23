declare -a ids=("04225987" "04460130" "04468005" "04530566")
poses=("forward" "top" "left" "right")
label=8
for id in "${ids[@]}"
do
	echo $id
	for objpose in "${poses[@]}"
	do
	    path=test_ids/$id/$objpose.txt
	    echo ${path}
	    python generate_images.py --id $id --hashcode_file ${path} --label $label --attack FGSM --params all --pose $objpose
	    python generate_images.py --id $id --hashcode_file ${path} --label $label --attack FGSM --params pose --pose $objpose
	    python generate_images.py --id $id --hashcode_file ${path} --label $label --attack FGSM --params vertex --pose $objpose
	done
	label=$((label + 1))
done;
