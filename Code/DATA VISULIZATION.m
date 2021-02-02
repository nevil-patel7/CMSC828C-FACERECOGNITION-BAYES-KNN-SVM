filename = 'data.mat';
images = load(filename);
faces = images.face;
image1 = faces(:,:,1);
image2 = faces(:,:,2);
image3 = faces(:,:,3);
image4 = faces(:,:,4);
image5 = faces(:,:,5);
image6 = faces(:,:,6);
image7 = faces(:,:,7);
image8 = faces(:,:,8);
image9 = faces(:,:,9);

subplot(3,3,1),
imshow(image1);
subplot(3,3,2),
imshow(image2)
subplot(3,3,3),
imshow(image3);

subplot(3,3,4),
imshow(image4);
subplot(3,3,5),
imshow(image5)
subplot(3,3,6),
imshow(image6);

subplot(3,3,7),
imshow(image7);
subplot(3,3,8),
imshow(image8)
subplot(3,3,9),
imshow(image9);

