
clear all;
[X,map] = imread('population-density-map.bmp');
image(X)

hold on

figure

[Y,map] = imread('elevation1x1_new-mer-bleue.bmp');
image(Y)

% [Z,map] = imread('elevation_wrt_population.bmp');
% image(Z)
fixedPoints4 =1.0e+03 *[2.9415    0.1331;
    0.6032    0.2795;
    3.7257    1.7180;
    0.2548    2.9895;
    0.2939    0.7020;
    1.0907    1.0444;
    0.5541    2.6874;
    2.0257    1.6896;
    0.6459    1.3111;
    0.2525    1.9029;
    1.7296    0.3553;
    2.5089    0.2243;
    4.0865    1.4346;
    4.4969    2.0406;
    3.2965    1.9472;
    4.1443    2.5879;
    3.9533    2.8395;
    3.0692    2.6930;
    2.3684    2.4178;
    1.2940    1.9651]
    
movingPoints4 = 1.0e+03 *[3.2082    1.3621;
    1.1841    1.2880;
    3.7288    3.0206;
    0.0504    3.8776;
    0.7787    1.6354;
    1.3533    2.2367;
    0.4171    3.7010;
    2.0770    3.0577;
    0.8786    2.3906;
    0.3474    2.8399;
    2.1109    1.6170;
    2.8202    1.5007;
    4.0897    2.6416;
    4.4486    3.1858;
    3.2881    3.3176;
    4.0821    3.8362;
    3.8734    4.1234;
    2.9765    4.0783;
    2.2902    3.8049;
    1.3063    3.2142]
    
    
fixedPoints =1.0e+03 *[2.9415    0.1331;
    0.6032    0.2795;
    3.9670    1.6381;
    0.2548    2.9895;
    0.2939    0.7020;
    1.0907    1.0444;
    0.5541    2.6874;
    2.1071    1.5893;
    0.6187    1.3019;
    0.2525    1.9029;
    1.7311    0.3541;
    2.5089    0.2243;
    4.0865    1.4346;
    4.4969    2.0406;
    3.2965    1.9472;
    4.1443    2.5879;
    3.9533    2.8395;
    3.0692    2.6930;
    2.3684    2.4178;
    1.2940    1.9651]

movingPoints =1.0e+03 *[3.2082    1.3621;
    1.1841    1.2880;
    3.9639    2.8909;
    0.0504    3.8776;
    0.7787    1.6354;
    1.3533    2.2367;
    0.4171    3.7010;
    2.1729    2.9614;
    0.8569    2.3707;
    0.3474    2.8399;
    2.1132    1.6147;
    2.8202    1.5007;
    4.0897    2.6416;
    4.4486    3.1858;
    3.2881    3.3176;
    4.0821    3.8362;
    3.8734    4.1234;
    2.9765    4.0783;
    2.2902    3.8049;
    1.3063    3.2142]

tform = fitgeotrans(movingPoints4,fixedPoints4,'polynomial',4)

Jregistered = imwarp(Y,tform,'OutputView',imref2d(size(X)));
figure
image(Jregistered)
figure
%image(Jregistered)
imshowpair(X,Jregistered)

imwrite(Jregistered, 'elevation_wrt_population.bmp');



