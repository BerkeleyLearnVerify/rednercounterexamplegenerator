declare -a ids=("02958343" "03513137" "03710193" "03790512")
poses=("forward" "top" "left" "right")
label=4
for id in "${ids[@]}"
do
	echo $id
	for objpose in "${poses[@]}"
	do 
	    path=dataset_splits/test_ids/$id/$objpose.txt
	    echo ${path}
	    python generate_images.py --id $id --hashcode_file ${path} --label $label --attack PGD --params all --pose $objpose 
	    python generate_images.py --id $id --hashcode_file ${path} --label $label --attack PGD --params pose --pose $objpose
	    python generate_images.py --id $id --hashcode_file ${path} --label $label --attack PGD --params vertex --pose $objpose
	done
	label=$((label + 1))
done;
