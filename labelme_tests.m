HOMEANNOTATIONS = 'http://labelme.csail.mit.edu/Annotations'; 
HOMEIMAGES = 'http://labelme.csail.mit.edu/Images'; 
D = LMdatabase(HOMEANNOTATIONS, {'static_street_statacenter_cambridge_outdoor_2005'});
LMdbshowscenes(D, HOMEIMAGES);

%%
% Define the root folder for the images
% HOMEIMAGES = 'E:\datasets\labelme-dataset\Images'; % you can set here your default folder
% HOMEANNOTATIONS = 'E:\datasets\labelme-dataset\Annotations'; % you can set here your default folder

% This line reads the entire database into a Matlab struct
HOMEANNOTATIONS = 'http://labelme.csail.mit.edu/Annotations'; 
HOMEIMAGES = 'http://labelme.csail.mit.edu/Images';

% Street scenes
% To get images that have trees, buildings and roads:
[D,j1] = LMquery(Dlabelme, 'object.name', 'building');
[D,j2] = LMquery(Dlabelme, 'object.name', 'road');
[D,j3] = LMquery(Dlabelme, 'object.name', 'tree');
j = intersect(intersect(j1,j2),j3);
LMdbshowscenes(LMquery(Dlabelme(j), 'object.name', 'car,building,road,tree'), HOMEIMAGES);

%% install full dataset
% Define the root folder for the images
HOMEIMAGES = 'E:\datasets\labelme-dataset\Images'; % you can set here your default folder
HOMEANNOTATIONS = 'E:\datasets\labelme-dataset\Annotations'; % you can set here your default folder

LMinstall (HOMEIMAGES, HOMEANNOTATIONS);

%%
HOMEIMAGES = 'http://people.csail.mit.edu/brussell/research/LabelMe/Images';
HOMEANNOTATIONS = 'http://people.csail.mit.edu/brussell/research/LabelMe/Annotations';

D = LMdatabase(HOMEANNOTATIONS); % This will build an index, which will take few minutes.

% Now you can visualize the images
LMplot(D, 1, HOMEIMAGES);

% Or read an image
[annotation, img] = LMread(D, 1, HOMEIMAGES);

% query and download outdoor images
% Street scenes
% To get images that have trees, buildings and roads:
[D,j1] = LMquery(D, 'object.name', 'building');
[D,j2] = LMquery(D, 'object.name', 'road');
[D,j3] = LMquery(D, 'object.name', 'tree');
[D,j4] = LMquery(D, 'object.name', 'outdoor');
[D,j5] = LMquery(D, 'object.name', 'car');

j = intersect(intersect(intersect(j1,j2),intersect(j3,j4)), j5);
LMdbshowscenes(LMquery(D(j), 'object.name', 'building,road,tree, car, outdoor'), HOMEIMAGES);

% First create the list of images that you want:

clear folderlist filelist
for i = 1:length(D);
      folderlist{i} = D(i).annotation.folder;
      filelist{i} = D(i).annotation.filename;
end

% Install the selected images:
HOMEIMAGES = 'E:\datasets\labelme-dataset\Images';
HOMEANNOTATIONS = 'E:\datasets\labelme-dataset\Annotations';
LMinstall (folderlist, filelist, HOMEIMAGES, HOMEANNOTATIONS);