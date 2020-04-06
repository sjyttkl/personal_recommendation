user_rating_file=
train_file=
item_vec_file=
item_sim_file=
if [-f $user_rating_file];then
    $python produce_train_data.py $user_rating_file $train_file
else
    echo "n rating file"
    exit(1)
fi
if [-f $train_file];then
    sh train.sh $train_file $item_vec_file
else
    echo "no trian file"
    exit(1)
fi
if[-f $item_vec_file]; then
    $python produce_item_sim.py $item_vec_file $item_sim_file
else
    eho "no item vec file"
fi
