%employing color domain processing (HSV)and texture Analysis for skin lesions classification
clc;
clear all;
close all;
cd data

df=[]

for i = 1:22
    x=imread(strcat(int2str(i),'.jpg'));
    
    %%%%%%%basic features%%%%%%%%
    x1=rgb2hsv(x);
    H=x1(:,:,1);
    S=x1(:,:,2);
    V=x1(:,:,3);
    
    Hm=mean(mean(H));
    Sm=mean(mean(S));
    Vm=mean(mean(V));
    
    Hsd=std2(H);
    Ssd=std2(S);
    Vsd=std2(V);
    
    Hsk=sum(skewness(H));
    Ssk=sum(skewness(S));
    Vsk=sum(skewness(V));
    
    Hmin=min(imhist(H));
    Smin=min(imhist(S));
    Vmin=min(imhist(V));
    
    Hmax=max(imhist(H));
    Smax=max(imhist(S));
    Vmax=max(imhist(V));
   
   %%%%%%%texture features%%%%%%%%
   
   y=rgb2gray(x);
   glcm=graycomatrix(y,'Offset',[2,0;0,2]);   %Gray-Level Co-Occurrence Matrix 
   st1=graycoprops(glcm,{'contrast','homogeneity'});
   st2=graycoprops(glcm,{'correlation','energy'});
   
   f1=st1.Contrast;
   f2=st1.Homogeneity;
   f3=st2.Correlation;
   f4=st2.Energy;
   
   Fftr=horzcat([Hm,Sm,Vm,Hsd,Ssd,Vsd,Hsk,Ssk,Vsk,Hmin,Smin,Vmin,Hmax,Smax,Vmax,f1,f2,f3,f4]);
   
   df=[df;Fftr];
     
end
    
cd ..

%%%%%%%%%%%%get test image %%%%%%%%%%
    [f,p]=uigetfile('*.*');
    test=imread(strcat(p,f));
    x1=rgb2hsv(test);
    H=x1(:,:,1);
    S=x1(:,:,2);
    V=x1(:,:,3);
    
    Hm=mean(mean(H));
    Sm=mean(mean(S));
    Vm=mean(mean(V));
    
    Hsd=std2(H);
    Ssd=std2(S);
    Vsd=std2(V);
    
    Hsk=sum(skewness(H));
    Ssk=sum(skewness(S));
    Vsk=sum(skewness(V));
    
    Hmin=min(imhist(H));
    Smin=min(imhist(S));
    Vmin=min(imhist(V));
    
    Hmax=max(imhist(H));
    Smax=max(imhist(S));
    Vmax=max(imhist(V));
   
   %%%%%%%texture features test image%%%%%%%%
   
   y=rgb2gray(test);
   glcm=graycomatrix(y,'Offset',[2,0;0,2]);   %Gray-Level Co-Occurrence Matrix 
   st1=graycoprops(glcm,{'contrast','homogeneity'});
   st2=graycoprops(glcm,{'correlation','energy'});
   
   f1=st1.Contrast;
   f2=st1.Homogeneity;
   f3=st2.Correlation;
   f4=st2.Energy;
  
   Testftr=horzcat([Hm,Sm,Vm,Hsd,Ssd,Vsd,Hsk,Ssk,Vsk,Hmin,Smin,Vmin,Hmax,Smax,Vmax,f1,f2,f3,f4]);
   
   %%%%%%%%%%%%%%%%%%%%%%%trainning
   TrainingSet=df;
   GroupTrain={'1','1','1','1','1','2','2','2','2','3','3','3','3','3','4','4','4','4','5','5','5','5'};
   
   TestSet=Testftr;
   
   
   %%%%%%%%%%SVM
   Y=GroupTrain;
   classes=unique(Y);
   SVMModels=cell(length(classes),1);
   rng(1);   %Reproductivity
   
   for j=1:numel(classes)
       idx=strcmp(Y',classes(j));
       SVMModels{j}=fitcsvm(df,idx,'ClassNames',[false true],'Standardize',true,'KernelFunction','rbf','BoxConstraint',1)

   end
   xGrid=Testftr;
   for j=1:numel(classes)
   [~,score]=predict(SVMModels{j},xGrid)
   Scores(:,j)=score(:,2);
   end
   
   
   [~,maxScore]=max(Scores,[],2)
   
   result=maxScore;
   
   if result == 1
       msgbox('Basal Cell carcinoma-skin lesion')
   elseif result == 2
       msgbox('Melanoma-Skin Lesion')
   elseif result == 3
       msgbox('Dermofit-Skin Lesion')
   elseif result == 4
       msgbox('Pigmented-Skin Lesion')
   elseif result == 5
       msgbox('Seborrheic Keratosis-Skin Lesion')
  
   else
       msgbox('None')
   end    
   
   
   
       
   
   
   
   
   
   
   
   





