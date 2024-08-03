clc;clear;close all;
tic;


%% Wifi Packet Paramters
LENGTH = 100;      % 1-4095
DataRate = [6,9,12,18,24,36,48,54];      % 6,9,12,18,24,36,48,54  --6,9,36,54 errors
ModOrder = [2,2,4,4,16,16,64,64];

%% Simulation paramters
MaxSNR =  -5;
SNR = -10:2:MaxSNR;
SNR_linear = 10.^(SNR/10);
Iterations = 5;

%% STO Paramters
Min_STO_Samples = 100;
Max_STO_Samples = 200;

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

added_Samples = randi([Min_STO_Samples,Max_STO_Samples]);
Effects.add_STO(added_Samples);
Effects.add_Noise(MaxSNR);

wavefor_STO = Effects.TransmitterOutput.waveform;

SignalPower = (wavefor_STO.*conj(wavefor_STO));

figure
plot( 1:length(wavefor_STO), SignalPower)
title("Signal Power")

threshold = 0.015;
signal_start = find(SignalPower>threshold);
signal_start = signal_start(1)
