% Read U-Net Output
Hpath = './UNettestOutput_nodp/'; % Change this to the folder containing the npy files
addpath('natsortfiles');
filenames = fullfile(Hpath,'*.npy');
Hfiles = natsortfiles(dir(filenames));
nvol = numel(Hfiles)/2;

% Label Numbers -> 1 - Background, 2 - ET, 3 - ED, 4 - NCR/NET

% 3 Regions
% Whole Tumour (WT) = All Labels (ET, ED, NCR/NET)
% Tumour Core (TC) = All Labels except Edema (ET, NCR/NET)
% Enhancing Tumour Region (ET) = (ET)

DSC = zeros(3,nvol); % Dice coefficients 
Sensitivity = zeros(3,nvol); % Sensitivity
Specificity = zeros(3,nvol); % Specificity
hd95_max = zeros(3,nvol);
region_str = {'Whole Tumour','Tumour Core','Enhancing Tumour'};


for region = 1:3 % loop through WT, TC, and ET
    ct = 1; 
    for vol = 1:2:nvol*2 % loop through volumes
    
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
            if reg == 2
                UNet_pred_reord(:,:,:,reg) = round(temp_reg);
            else
                UNet_pred_reord(:,:,:,reg) = round(temp_reg);
            end
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
        
        P_1_s = sum(UNet_pred(:));
        T_1_s = sum(UNet_GT(:));
        T_0_s = numel(UNet_GT) - sum(UNet_GT(:));
   
        P_1_and_T_1 = double(UNet_pred & UNet_GT); 
        P_1_and_T_1_s = sum(P_1_and_T_1(:));
    
        P_0_and_T_0 = double((UNet_pred == 0) & (UNet_GT == 0)); 
        P_0_and_T_0_s = sum(P_0_and_T_0(:));
        
        DSC(region,ct) = 2*P_1_and_T_1_s/(P_1_s + T_1_s); % Dice Coefficient
        Sensitivity(region,ct) = P_1_and_T_1_s/T_1_s*100; % Sensitivity
        Specificity(region,ct) = P_0_and_T_0_s/T_0_s*100; % Specificity
    
        volname = split(Hfiles(vol).name,'_'); 
        % Calculate Hausdoff Distance (95% percentile)
        hd95 = zeros(1,155);
        parfor slice = 1:155
           hd95(slice) =  HausdorffDist(reshape(UNet_pred(slice,:,:),[240 240]),reshape(UNet_GT(slice,:,:),[240 240]));
        end
        hd95_max(region,ct) = max(hd95(:));
    
        disp([region_str{region},': ',volname{1},' DSC = ',num2str(DSC(region,ct)), ', Sensitivity = ',num2str(Sensitivity(region,ct)), ...
            '%, ', 'Specificity = ', num2str(Specificity(region,ct)), '%', ', HD95 = ',num2str(hd95_max(region,ct)), 'mm'])
    
        ct = ct + 1;
    
    end

    % Display the average +/- standard deviation
    disp([region_str{region},': Avg DSC = ',num2str(mean(DSC(region,:))), '+/-',num2str(std(DSC(region,:))), ...
    ', Avg Sensitivity = ',num2str(mean(Sensitivity(region,:))), '+/-',num2str(std(Sensitivity(region,:))),'%, ', ...
    'Avg Specificity = ', num2str(mean(Specificity(region,:))),  '+/-',num2str(std(Specificity(region,:))), '%', ... 
    ', Avg HD95 = ',num2str(mean(hd95_max(region,:))), '+/-',num2str(std(hd95_max(region,:))), 'mm'])
end

%% Boxplots

% Dice-Coefficient
figure (); 
boxplot(DSC', 'positions', 1:3, 'Colors','k','Symbol','w'); hold on; 
%ylabel('%')
set(gca,'xticklabel',{'WT','TC','ET'})
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'FontName', 'Arial','FontSize',12)
hold on; 
colour_list = ['b','g','r'];
for k = 1:3
    scatter(k.*ones(1,nvol), DSC(k,:), 20, colour_list(k), 'filled'); hold on;
end
title('Dice Sorenson Coefficient')

% Sensitivity
figure (); 
boxplot(Sensitivity', 'positions', 1:3, 'Colors','k','Symbol','w'); hold on; 
ylabel('%')
set(gca,'xticklabel',{'WT','TC','ET'})
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'FontName', 'Arial','FontSize',12)
hold on; 
colour_list = ['b','g','r'];
for k = 1:3
    scatter(k.*ones(1,nvol), Sensitivity(k,:), 20, colour_list(k), 'filled'); hold on;
end
title('Sensitivity')

% Specificity
figure (); 
boxplot(Specificity', 'positions', 1:3, 'Colors','k','Symbol','w'); hold on; 
ylabel('%')
set(gca,'xticklabel',{'WT','TC','ET'})
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'FontName', 'Arial','FontSize',12)
hold on; 
colour_list = ['b','g','r'];
for k = 1:3
    scatter(k.*ones(1,nvol), Specificity(k,:), 20, colour_list(k), 'filled'); hold on;
end
title('Specificity')

% Hausdorff 95% percentile
figure (); 
boxplot(hd95_max', 'positions', 1:3, 'Colors','k','Symbol','w'); hold on; 
ylabel('mm')
set(gca,'xticklabel',{'WT','TC','ET'})
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'FontName', 'Arial','FontSize',12)
hold on; 
colour_list = ['b','g','r'];
for k = 1:3
    scatter(k.*ones(1,nvol), hd95_max(k,:), 20, colour_list(k), 'filled'); hold on;
end
title('Hausdorff 95% percentile')
