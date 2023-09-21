python spmotif_student.py \
  --dataset spmotif \
  --model_name=Graph_Student_MSE \
  --by_default \
  --gnn=gcn-virtual \
  --device=8 \
  --trails=5 \
  --beta_infonce=0.01 \
  --beta_club=0.01 \
  --train_type=student \
  --teacher_model=./spmotif0.9_gcn_teacher_1 \




