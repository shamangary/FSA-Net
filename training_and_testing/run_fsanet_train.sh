# Train on protocal 1
# SSRNET_MT
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 8 --nb_epochs 90 --model_type 0 --db_name '300W_LP'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 8 --nb_epochs 90 --model_type 1 --db_name '300W_LP'

# FSANET_Capsule
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 2 --db_name '300W_LP'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 3 --db_name '300W_LP'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 4 --db_name '300W_LP'


# FSANET_Netvlad
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 5 --db_name '300W_LP'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 6 --db_name '300W_LP'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 7 --db_name '300W_LP'


# FSANET_Metric
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 8 --db_name '300W_LP'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 9 --db_name '300W_LP'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 10 --db_name '300W_LP'


# ===============================================================================================================
# Train on protocal 2

# SSRNET_MT
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 0 --db_name 'BIWI'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 1 --db_name 'BIWI'

# FSANET_Capsule
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 2 --db_name 'BIWI'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 3 --db_name 'BIWI'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 4 --db_name 'BIWI'


# FSANET_Netvlad
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 5 --db_name 'BIWI'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 6 --db_name 'BIWI'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 7 --db_name 'BIWI'


# FSANET_Metric
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 8 --db_name 'BIWI'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 9 --db_name 'BIWI'
KERAS_BACKEND=tensorflow python FSANET_train.py --batch_size 16 --nb_epochs 90 --model_type 10 --db_name 'BIWI'




# ===============================================================================================================

# Fine-tuned on BIWI with synhead pre-trained model
KERAS_BACKEND=tensorflow python FSANET_fine_train.py --batch_size 8 --nb_epochs 90 --model_type 1 --db_name 'BIWI'
KERAS_BACKEND=tensorflow python FSANET_fine_train.py --batch_size 8 --nb_epochs 90 --model_type 2 --db_name 'BIWI'
KERAS_BACKEND=tensorflow python FSANET_fine_train.py --batch_size 8 --nb_epochs 90 --model_type 3 --db_name 'BIWI'
