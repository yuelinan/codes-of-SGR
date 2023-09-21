python sst_student.py \
  --dataset sst \
  --model_name=Graph_Student_MSE \
  --by_default \
  --gnn=gin-virtual \
  --device=5 \
  --trails=1 \
  --beta_infonce=0.01 \
  --beta_club=0.01 \
  --train_type=student \
  --teacher_model=./gin_sst_teacher_2 \

  