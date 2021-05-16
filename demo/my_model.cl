/*
OpenCL RandomForestClassifier
feature_specification = original gaussian_blur=2 sobel_of_gaussian_blur=2num_classes = 3
num_features = 3
max_depth = 2
num_trees = 10
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_out_TYPE out) {
 sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 const int x = get_global_id(0);
 const int y = get_global_id(1);
 const int z = get_global_id(2);
 float i0 = READ_IMAGE(in0, sampler, POS_in0_INSTANCE(x,y,z,0)).x;
 float i1 = READ_IMAGE(in1, sampler, POS_in1_INSTANCE(x,y,z,0)).x;
 float i2 = READ_IMAGE(in2, sampler, POS_in2_INSTANCE(x,y,z,0)).x;
 float s0=0;
 float s1=0;
 float s2=0;
if(i2<122.18116760253906){
 if(i0<180.0){
  s0+=239.0;
  s2+=32.0;
 } else {
  s1+=122.0;
  s2+=1.0;
 }
} else {
 if(i0<212.0){
  s0+=84.0;
  s1+=3.0;
  s2+=386.0;
 } else {
  s1+=7.0;
 }
}
if(i0<196.0){
 if(i2<157.8175048828125){
  s0+=267.0;
  s2+=89.0;
 } else {
  s0+=22.0;
  s2+=328.0;
 }
} else {
 if(i1<187.36000061035156){
  s2+=22.0;
 } else {
  s1+=146.0;
 }
}
if(i0<204.0){
 if(i2<167.7120361328125){
  s0+=312.0;
  s1+=7.0;
  s2+=127.0;
 } else {
  s0+=15.0;
  s2+=259.0;
 }
} else {
 if(i1<189.14828491210938){
  s2+=9.0;
 } else {
  s1+=145.0;
 }
}
if(i1<187.36000061035156){
 if(i2<137.2589569091797){
  s0+=241.0;
  s2+=51.0;
 } else {
  s0+=70.0;
  s2+=377.0;
 }
} else {
 s1+=135.0;
}
if(i0<204.0){
 if(i2<158.2269744873047){
  s0+=301.0;
  s1+=9.0;
  s2+=97.0;
 } else {
  s0+=23.0;
  s2+=299.0;
 }
} else {
 if(i0<220.0){
  s1+=30.0;
  s2+=14.0;
 } else {
  s1+=101.0;
 }
}
if(i1<189.14828491210938){
 if(i2<141.07102966308594){
  s0+=254.0;
  s2+=54.0;
 } else {
  s0+=68.0;
  s2+=365.0;
 }
} else {
 s1+=133.0;
}
if(i2<157.8175048828125){
 if(i1<185.8437957763672){
  s0+=309.0;
  s2+=87.0;
 } else {
  s1+=140.0;
 }
} else {
 if(i1<189.70053100585938){
  s0+=33.0;
  s2+=300.0;
 } else {
  s1+=5.0;
 }
}
if(i1<189.14828491210938){
 if(i1<100.63290405273438){
  s0+=288.0;
  s2+=139.0;
 } else {
  s0+=36.0;
  s2+=268.0;
 }
} else {
 s1+=143.0;
}
if(i1<189.14828491210938){
 if(i2<158.2269744873047){
  s0+=336.0;
  s2+=89.0;
 } else {
  s0+=20.0;
  s2+=290.0;
 }
} else {
 s1+=139.0;
}
if(i0<204.0){
 if(i0<92.0){
  s0+=266.0;
  s2+=147.0;
 } else {
  s0+=43.0;
  s1+=5.0;
  s2+=261.0;
 }
} else {
 if(i0<212.0){
  s1+=25.0;
  s2+=8.0;
 } else {
  s1+=117.0;
  s2+=2.0;
 }
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 if (max_s < s2) {
  max_s = s2;
  cls=3;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}
