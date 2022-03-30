% Read U-Net Output

for model = 1:2 % Loop through the U-Net and U-Net++

    if model == 1
        Hpath = 'AI Project/UNETtestOutput_nodp/'; % Change this to the folder containing the npy files
    else 
        Hpath = 'AI Project/UNETPPtestOutput_nodp/'; % Change this to the folder containing the npy files
    end

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
    
    if model == 1
        DSC_UNet = DSC; 
        Sensitivity_UNet = Sensitivity; 
        Specificity_UNet = Specificity;
        hd95_max_UNet = hd95_max; 
    else
        DSC_UNetP = DSC; 
        Sensitivity_UNetP = Sensitivity; 
        Specificity_UNetP = Specificity;
        hd95_max_UNetP = hd95_max; 
    end

end
%% Boxplots 

% Dice-Coefficient
DSC_full = [DSC_UNet(1,:)', DSC_UNetP(1,:)', DSC_UNet(2,:)', DSC_UNetP(2,:)', DSC_UNet(3,:)', DSC_UNetP(3,:)' ];
positions = [1.25 1.75 2.25 2.75 3.25 3.75];
figure (); 
boxplot(DSC_full, 'positions',positions, 'Colors','k','Symbol','w'); hold on; 
set(gca,'xtick',[mean(positions(1:2)) mean(positions(3:4)) mean(positions(5:6))])
set(gca,'xticklabel',{'WT','TC','ET'})
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'FontName', 'Arial','FontSize',12)
hold on; 
colour_list = ['b','g','r'];
for k = 1:3
    scatter((k+.25).*ones(1,nvol), DSC_full(:,k), 20, colour_list(k), 'filled'); hold on;
    scatter((k+.75).*ones(1,nvol), DSC_full(:,k+1), 20, colour_list(k), '^','filled'); hold on;
end
legend('WT - UNet','WT - UNet++','TC - UNet','TC - UNet++','ET - UNet','ET - UNet++')
title('DSC')

% Test Significance
for k = 1:3
    [h,p] = ttest(DSC_UNet(k,:),DSC_UNetP(k,:));
    fprintf(['The p-value is ', num2str(p), ' and the h-value is ', num2str(h), '\n'])
end
%% Sensitivity
Sensitivity_full = [Sensitivity_UNet(1,:)', Sensitivity_UNetP(1,:)', Sensitivity_UNet(2,:)', Sensitivity_UNetP(2,:)', Sensitivity_UNet(3,:)', Sensitivity_UNetP(3,:)' ];
positions = [1.25 1.75 2.25 2.75 3.25 3.75];
figure (); 
boxplot(Sensitivity_full, 'positions',positions, 'Colors','k','Symbol','w'); hold on; 
set(gca,'xtick',[mean(positions(1:2)) mean(positions(3:4)) mean(positions(5:6))])
set(gca,'xticklabel',{'WT','TC','ET'})
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'FontName', 'Arial','FontSize',12)
hold on; 
colour_list = ['b','g','r'];
for k = 1:3
    scatter((k+.25).*ones(1,nvol), Sensitivity_full(:,k), 20, colour_list(k), 'filled'); hold on;
    scatter((k+.75).*ones(1,nvol), Sensitivity_full(:,k+1), 20, colour_list(k), '^','filled'); hold on;
end
%legend('WT - UNet','WT - UNet++','TC - UNet','TC - UNet++','ET - UNet','ET - UNet++')
title('Sensitivity (%)')

% Test Significance
for k = 1:3
    [h,p] = ttest(Sensitivity_UNet(k,:),Sensitivity_UNetP(k,:));
    fprintf(['The p-value is ', num2str(p), ' and the h-value is ', num2str(h), '\n'])
end
%% Specificity
Specificity_full = [Specificity_UNet(1,:)', Specificity_UNetP(1,:)', Specificity_UNet(2,:)', Specificity_UNetP(2,:)', Specificity_UNet(3,:)', Specificity_UNetP(3,:)' ];
positions = [1.25 1.75 2.25 2.75 3.25 3.75];
figure (); 
boxplot(Specificity_full, 'positions',positions, 'Colors','k','Symbol','w'); hold on; 
set(gca,'xtick',[mean(positions(1:2)) mean(positions(3:4)) mean(positions(5:6))])
set(gca,'xticklabel',{'WT','TC','ET'})
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'FontName', 'Arial','FontSize',12)
hold on; 
colour_list = ['b','g','r'];
for k = 1:3
    scatter((k+.25).*ones(1,nvol), Specificity_full(:,k), 20, colour_list(k), 'filled'); hold on;
    scatter((k+.75).*ones(1,nvol), Specificity_full(:,k+1), 20, colour_list(k), '^','filled'); hold on;
end
%legend('WT - UNet','WT - UNet++','TC - UNet','TC - UNet++','ET - UNet','ET - UNet++')
title('Specificity (%)')

% Test Significance
for k = 1:3
    [h,p] = ttest(Specificity_UNet(k,:),Specificity_UNetP(k,:));
    fprintf(['The p-value is ', num2str(p), ' and the h-value is ', num2str(h), '\n'])
end
%% Hausdorff 95% percentile
figure (); 
hd95_max_full = [hd95_max_UNet(1,:)', hd95_max_UNetP(1,:)', hd95_max_UNet(2,:)', hd95_max_UNetP(2,:)', hd95_max_UNet(3,:)', hd95_max_UNetP(3,:)' ];
positions = [1.25 1.75 2.25 2.75 3.25 3.75];
figure (); 
boxplot(hd95_max_full, 'positions',positions, 'Colors','k','Symbol','w'); hold on; 
set(gca,'xtick',[mean(positions(1:2)) mean(positions(3:4)) mean(positions(5:6))])
set(gca,'xticklabel',{'WT','TC','ET'})
set(gca,'TickLabelInterpreter', 'tex');
set(gca, 'FontName', 'Arial','FontSize',12)
hold on; 
colour_list = ['b','g','r'];
for k = 1:3
    scatter((k+.25).*ones(1,nvol), hd95_max_full(:,k), 20, colour_list(k), 'filled'); hold on;
    scatter((k+.75).*ones(1,nvol), hd95_max_full(:,k+1), 20, colour_list(k), '^','filled'); hold on;
end
title('HD95 (mm)')

% Test Significance
for k = 1:3
    [h,p] = ttest(hd95_max_UNet(k,:),hd95_max_UNetP(k,:));
    fprintf(['The p-value is ', num2str(p), ' and the h-value is ', num2str(h), '\n'])
end
