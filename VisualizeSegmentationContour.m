% Analysis of the U-Net Ouput 
Vpath = 'BraTS2020_training_data/BraTS2020_training_data/content/data';
outV = zeros(240,240,155,4); 
outL = zeros(240,240,155,3); 

for id = 3
    for slicen = 0:154 
        filename = [Vpath,'/volume_',num2str(id),'_slice_',num2str(slicen),'.h5'];
        tempV = h5read(filename,'/image');
        for mm = 1:4 % 4 channels
            placeh = reshape(permute(tempV(mm,:,:),[3 2 1]),[240 240]);
            outV(:,:,slicen+1,mm) = placeh;
        end

        tempL = h5read(filename,'/mask');
        %outL(:,:,slicen+1) = permute(double(sum(double(tempL)) > 0),[3 2 1]); 

        for mm = 1:3 % 3 labels - ET, ED, NCR/NET
            placeh = reshape(permute(tempL(mm,:,:),[3 2 1]),[240 240]);
            outL(:,:,slicen+1,mm) = placeh;
        end
        
    end
end

%% Example Overlay
for slice = 74 % choose a slice
    slice_temp = outV(:,:,slice ,4); 
    mask_temp = sum(outL(:,:,:,1:3),4);
    mask_temp = double(mask_temp(:,:,slice) > 0);
    
    figure ();
    imagesc(slice_temp);axis off, axis square; colormap gray;
    hold on, contour(mask_temp,[0 1],'g','LineWidth',1);
end

%% Read U-Net Output
% Read U-Net Output
Hpath = 'AI Project/UNETtestOutput_nodp/'; % Change this to the folder containing the npy files
addpath('natsortfiles');
filenames = fullfile(Hpath,'*.npy');
Hfiles = natsortfiles(dir(filenames));

% Label Numbers -> 1 - Background, 2 - ET, 3 - ED, 4 - NCR/NET

% 3 Regions
% Whole Tumour (WT) = All Labels (ET, ED, NCR/NET)
% Tumour Core (TC) = All Labels except Edema (ET, NCR/NET)
% Enhancing Tumour Region (ET) = (ET)

for region = 3 % loop through WT, TC, and ET 
    for vol = 1 % loop through volumes
    
        UNet_GT = readNPY([Hpath,Hfiles(vol).name]);
        UNet_GT_reord = zeros(155,240,240,4);
        
        for reg = 1:4
            temp_reg = reshape(UNet_GT(:,reg,:,:),[155 240 240]);
            UNet_GT_reord(:,:,:,reg) = temp_reg;
        end
       
        UNet_pred = readNPY([Hpath,Hfiles(vol+1).name]);   
        UNet_pred_reord = zeros(155,240,240,4);
        
        for reg = 1:4
            temp_reg = reshape(UNet_pred(:,reg,:,:),[155 240 240]);
            UNet_pred_reord(:,:,:,reg) = round(temp_reg);
        end
        
        % Ground Truth
        if region == 1
            temp = sum(UNet_GT_reord(:,:,:,2:4),4); % Whole Tumour (ET, ED, NCR/NET)
        elseif region == 2 
            temp = sum(UNet_GT_reord(:,:,:,[2,4]),4); % Tumour Core (ET, NCR/NET)
        else
            temp = UNet_GT_reord(:,:,:,2); % Enhancing Tumour (ET)
        end
        UNet_GT = double(temp > 0);
    
        % Predicted Labels
        if region == 1
            tempP = sum(UNet_pred_reord(:,:,:,2:4),4); % Whole Tumour (ET, ED, NCR/NET)
        elseif region == 2 
            tempP = sum(UNet_pred_reord(:,:,:,[2,4]),4); % Tumour Core (ET, NCR/NET)
        else
            tempP = UNet_pred_reord(:,:,:,2); % Enhancing Tumour (ET)
        end
        UNet_pred = double(tempP > 0);
    
    end

end


% Example Overlay
% close all;
for slice = 74 % choose a slice
    slice_temp = outV(:,:,slice,4); 
    GTcon = reshape(UNet_GT(slice,:,:),[240 240]);
    Predcon = reshape(UNet_pred(slice,:,:),[240 240]);
    
    figure ();
    imagesc(slice_temp);axis off, axis square; colormap gray;
    hold on, contour(GTcon,[0 1],'g','LineWidth',1);
    hold on, contour(Predcon,[0 1],'m','LineWidth',1);
end