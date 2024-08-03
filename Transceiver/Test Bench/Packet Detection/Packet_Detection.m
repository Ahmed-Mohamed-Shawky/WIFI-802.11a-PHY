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

%% Channel Paramters
Max_Delay_Spread = 200; % ns
%% CFO Paramters
Ratio = 100; %% (0-100)% CFO Ratio Effect
%% STO Paramters
Min_STO_Samples = 100;
Max_STO_Samples = 200;

%% Data Generating
data_hex = randi(255,LENGTH,1);
data_bits = dec2bin(data_hex)-'0';

%% Parameters

SignalDetection_threshold = 0.015;  % signal detection threshold
packetDetectionMode = 'Cross'; % 'Auto'or'Cross' ->corrlation type
PacketDetection_threshold = 0.6;


%% Waveform Generating
% Creat Transmiter Object
Transmitter = IEEE802_11a_Transmitter(LENGTH);
% Generate Waveform
Wifi_Output = Transmitter.GenerateWaveform(data_hex);
%% Effect Creation
Effects = IEEE802_11a_Effects(Wifi_Output);
Effects.DebugMode = 1; % Enable Effects Debug Mode

%% Adding Effects
added_Samples = randi([Min_STO_Samples,Max_STO_Samples]);
% Effects.add_STO(added_Samples);
% Effects.add_Noise(MaxSNR);

Effects.add_CFO(Ratio);
Effects.add_Channel(Max_Delay_Spread,'Rayleigh');
added_Samples = randi([Min_STO_Samples,Max_STO_Samples]);
Effects.add_STO(added_Samples);
Effects.add_Noise(MaxSNR);

%% Signal Detection

waveform_STO = Effects.TransmitterOutput.waveform;

SignalPower = (waveform_STO.*conj(waveform_STO));

signal_start = find(SignalPower>SignalDetection_threshold);
signal_start = signal_start(1)

RxWaveform = waveform_STO(signal_start:end);

figure
subplot(2,1,1)
plot( 1:length(SignalPower), SignalPower)
title("Signal Power")
subplot(2,1,2)
plot( 1:length(RxWaveform), abs(RxWaveform))
title("Rx Waveform")


%% Packet Detection
switch(packetDetectionMode)
    case 'Auto'
        
    case 'Cross'
        STS_Waveform = shortPreamble2waveform();
        STS_16Sample = STS_Waveform(1:16);

        corrResult = xcorr(STS_16Sample,flip(RxWaveform(1:200)));
        corrResult = abs(corrResult)/max(abs(corrResult));

        peacks = find(corrResult>PacketDetection_threshold);

        if(sum(peacks)>3)
            disp("Packet Detected")
        else
            disp("Not A Packet")
        end

        figure
        subplot(2,1,1)
        plot( 1:length(RxWaveform(1:200)), abs(RxWaveform(1:200)))
        subplot(2,1,2)
        plot( 1:length(corrResult), corrResult)
        title("Corr result")

end







%% Functions

function shortPreambleWaveform = shortPreamble2waveform()
%% Short Preample Waveform
% Short preable sequance freq domain
shortPreambleSequance = sqrt(13/6) * [0, 0, 1+1i, 0, 0, 0, -1-1i, 0, 0, 0, 1+1i, 0, 0, 0, -1-1i, 0, 0, 0, -1-1i, 0, 0, 0, 1+1i, 0, 0, 0, 0,...
0, 0, 0, -1-1i, 0, 0, 0, -1-1i, 0, 0, 0, 1+1i, 0, 0, 0, 1+1i, 0, 0, 0, 1+1i, 0, 0, 0, 1+1i, 0,0].';
shortPreambleFreqDomain = [zeros(6,1);shortPreambleSequance;zeros(5,1)];

% short preamble sequance time domain
shortPreambleFreqDomainCShift = circshift(shortPreambleFreqDomain,64/2);
shortPreambleTimeDomain = ifft(shortPreambleFreqDomainCShift);
shortPreambleWaveform = [shortPreambleTimeDomain;shortPreambleTimeDomain;shortPreambleTimeDomain(1:32)];

end
