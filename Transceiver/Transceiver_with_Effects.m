clc;clear;close all;
tic;

% scales
mega=1e6;
nano=1e-9;
%% Wifi Packet Paramters
LENGTH = 100;      % 1-4095
DataRate = [6,9,12,18,24,36,48,54];      % 6,9,12,18,24,36,48,54  --6,9,36,54 errors
ModOrder = [2,2,4,4,16,16,64,64];

%% Simulation paramters
MaxSNR =  -10;
SNR = -10:2:MaxSNR;
SNR_linear = 10.^(SNR/10);
Iterations = 5;

%% Channel Paramters
Max_Delay_Spread = 200; % ns
%% CFO Paramters
Ratio = 2; %% (0-100)% CFO Ratio Effect
%% STO Paramters
Min_STO_Samples = 100;
Max_STO_Samples = 200;

%% Reciever Paramters
% % sync paramters
% PacketDetection_Threshod = 0.5;  % for Packet Detection
% NumberOfShortPreambles = 1; % 1-10
% TimeSync_Threshod = 0.34; % for Time Sync 
%% Debuging Modes

% 'Noise' , 'Channel' , 'Channel+Noise' , 'CFO' , 'CFO+Noise' , 
% 'Paket Shift', 'Paket Shift+Noise' , 'Paket Shift+CFO+Noise' ,
% 'Paket Shift+Channel+Noise' , 'All'
% SimulationType = 'Paket Shift+Channel+Noise';
SimulationType = 'CFO';
SimulateMode = 0;   % 0 -> One Simulation , 1 -> BER Simulation

% TX_DebugMode = 0;
Effects_DebugMode = 1;
% CFO_Mode = 1;      % Turn CFO On or Off
% TimeSyncMode = 0;  % Turn Time Syncronization On Off
% EqualizerMode = 1; % Turn Equalization On or Off
% PacketDetectionMode = 0; % Turn Packet Detection On or Off
% ConstlationPlot = 1; % Turn Ploting On or Off
% Tracking = 0;       % Turn Tracking On or Off
% Tracking_Mode = 0;  % 0 -> Tracking Using CP , 1 -> Tracking Using Pilots
% 
% PacketDetection = struct("DetectionMode",PacketDetectionMode, ...
%                           "NumberOfShortPreambles",NumberOfShortPreambles, ...
%                           "Detection_Threshod",PacketDetection_Threshod, ...
%                           "TimeSync_Threshod",TimeSync_Threshod);
% 
% RX_State = struct("DebugMode",debugMode, ...
%                   "PacketDetection",PacketDetection, ...
%                   "TimeSyncMode",TimeSyncMode,...
%                   "EqualizerMode",EqualizerMode, ...
%                   "Equalizer",'ZF', ...    %  'ZF' , 'MSSE'
%                   "ConstlationPlot",ConstlationPlot, ...
%                   "CFO_Mode",CFO_Mode, ...
%                   "Tracking",Tracking, ...
%                   "Tracking_Mode",Tracking_Mode, ...
%                   "SNR_linear",10.^(MaxSNR/10));

%% Simulation
if ~SimulateMode
%% Data Generating
data_hex = randi(255,LENGTH,1);
data_bits = dec2bin(data_hex)-'0';

%% Waveform Generating
% Creat Transmiter Object
Transmitter = IEEE802_11a_Transmitter(LENGTH);
% Generate Waveform
Wifi_Output = Transmitter.GenerateWaveform(data_hex);
%% Effect Creation
Effects = IEEE802_11a_Effects(Wifi_Output);
Effects.DebugMode = 1; % Enable Effects Debug Mode

%% Simulation
switch SimulationType
    case 'Noise'
        Effects.add_Noise(MaxSNR);
    case 'Channel'
        Effects.add_Channel(Max_Delay_Spread,'Rayleigh'); %Racian
    case 'Channel+Noise'
        Effects.add_Channel(Max_Delay_Spread,'Rayleigh');
        Effects.add_Noise(MaxSNR);
    case 'CFO'
        % Wifi_Output.waveform = Wifi_Output.waveform .* phase_shift.';
        Effects.add_CFO(Ratio);
    case 'CFO+Noise'
        Effects.add_CFO(Ratio);
        Effects.add_Noise(MaxSNR);
    case 'Paket Shift'
        added_Samples = randi([Min_STO_Samples,Max_STO_Samples]);
        Effects.add_STO(added_Samples);
    case 'Paket Shift+Noise'
        added_Samples = randi([Min_STO_Samples,Max_STO_Samples]);
        Effects.add_STO(added_Samples);
        Effects.add_Noise(MaxSNR);
    case 'Paket Shift+CFO+Noise'
        Effects.add_CFO(Ratio);
        added_Samples = randi([Min_STO_Samples,Max_STO_Samples]);
        Effects.add_STO(added_Samples);
        Effects.add_Noise(MaxSNR);
    case 'Paket Shift+Channel+Noise'
        Effects.add_Channel(Max_Delay_Spread,'Rayleigh');
        added_Samples = randi([Min_STO_Samples,Max_STO_Samples]);
        Effects.add_STO(added_Samples);
        Effects.add_Noise(MaxSNR);
    case 'All'
        Effects.add_Channel(Max_Delay_Spread,'Rayleigh');
        Effects.add_CFO(Ratio);
        added_Samples = randi([Min_STO_Samples,Max_STO_Samples]);
        Effects.add_STO(added_Samples);
        Effects.add_Noise(MaxSNR);
end

RX_State.SNR_linear = SNR_linear(end);

%% ZF 
% RX_State.Equalizer = "ZF";
% ZF_RX_data = General_Receiver_Function(Wifi_Output,RX_State);
% 
% ZF_RX_data_bits = dec2bin(ZF_RX_data)-'0';
% ZF_ByteError = sum(ZF_RX_data ~= data_hex)
% ZF_BitError = sum(sum(ZF_RX_data_bits ~= data_bits))/(LENGTH*8)

%% MMSE
% RX_State.Equalizer = "MMSE";
% MMSE_RX_data = General_Receiver_Function(Wifi_Output,RX_State);
% 
% MMSE_RX_data_bits = dec2bin(MMSE_RX_data)-'0';
% MMSE_ByteError = sum(MMSE_RX_data ~= data_hex)
% MMSE_BitError = sum(sum(MMSE_RX_data_bits ~= data_bits))/(LENGTH*8)

end

% %% BER Simulation
% if SimulateMode
% figure();
% for dataRateIndex = 1:length(DataRate)
%     dataRate = DataRate(dataRateIndex);
%     disp("Data Rate: ");disp(dataRate);
%     ByteError_SNR = zeros(length(SNR),1);
%     BitError_SNR = zeros(length(SNR),1);
%     for snr_index = 1:length(SNR)
%         disp("SNR: ");disp(SNR(snr_index));
%         ByteError = zeros(Iterations,1);
%         BitError = zeros(Iterations,1);
%         PacketError = 0;
%         for I = 1:Iterations
%             %% Data Generation
%             data_hex = randi(255,LENGTH,1);
%             data_bits = dec2bin(data_hex)-'0';
%             %% Creating Waveform
%             TranmitterOutput = General_Transmitter_Function(data_hex,LENGTH,dataRate);
%             %% Adding Efficts
%             switch SimulationType
%                 case 'Noise'
%                     TranmitterOutput.waveform = adding_Noise(TranmitterOutput.waveform,SNR_linear(snr_index));
%                 case 'Channel'
%                     TranmitterOutput.waveform = rayleighchan(TranmitterOutput.waveform);
%                 case 'Channel+Noise'
%                     TranmitterOutput.waveform = rayleighchan(TranmitterOutput.waveform);
%                     TranmitterOutput.waveform = adding_Noise(TranmitterOutput.waveform,SNR_linear(snr_index));
%                 % case 'CFO'
% 
%             end
%             % Waveform_Noise = Waveform;
%             %% Receiver Block
%             try
%                 RX_data = General_Receiver_Function(TranmitterOutput,RX_State);
%                 % Claculte BER
%                 RX_data_bits = dec2bin(RX_data)-'0';
%                 ByteError(I) = sum(RX_data ~= data_hex)/LENGTH;
%                 BitError(I) = sum(sum(RX_data_bits ~= data_bits))/(LENGTH*8);
%             catch
%                 RX_data = zeros(LENGTH,1);
%                 PacketError = PacketError+1;
% 
%                 RX_data_bits = dec2bin(RX_data)-'0';
%                 ByteError(I) = sum(RX_data ~= data_hex)/LENGTH;
%                 BitError(I) = sum(sum(RX_data_bits ~= data_bits))/(LENGTH*8);
%             end
% 
%         end
%         ByteError_SNR(snr_index) = sum(ByteError)/Iterations;
%         disp("Byte Error: ");disp(ByteError_SNR(snr_index));
%         BitError_SNR(snr_index) = sum(BitError)/Iterations;
%         disp("Bit Error: ");disp(BitError_SNR(snr_index));
%     end
%     hold on;
%     semilogy(SNR,BitError_SNR);
% end
%     title(strcat(SimulationType,' BER'))
%     xlabel('SNR')
%     ylabel('Bit Error')
%     legend("Data Rate: 6", ...
%            "Data Rate: 9", ...
%            "Data Rate: 12", ...
%            "Data Rate: 18", ...
%            "Data Rate: 24", ...
%            "Data Rate: 36", ...
%            "Data Rate: 48", ...
%            "Data Rate: 54");
% end
    toc;
% ByteError = sum(RX_data ~= data_hex)/LENGTH
% BitError = sum(sum(RX_data_bits ~= data_bits))/(LENGTH*8)