clc;clear;close all;
tic;

%% Wifi Packet Paramters
LENGTH = 100;      % 1-4095
DataRate = [6,9,12,18,24,36,48,54];      % 6,9,12,18,24,36,48,54  --6,9,36,54 errors
ModOrder = [2,2,4,4,16,16,64,64];

%% Simulation paramters
MaxSNR = 10;
SNR = -10:1:MaxSNR;
SNR_linear = 10.^(SNR/10);
Iterations = 5;

   
%% Data Generating
data_hex = randi(255,LENGTH,1);
data_bits = dec2bin(data_hex)-'0';

%% Waveform Generating
% Creat Transmiter Object
Transmitter = IEEE802_11a_Transmitter(LENGTH);
Transmitter.DebugMode = 1;

% Generate Waveform
TX_Output = Transmitter.GenerateWaveform(data_hex);

%% Adding Multipath Channel effect
Effects = IEEE802_11a_Effects(TX_Output);
Effects.DebugMode = 1; % Enable Effects Debug Mode

Effects.add_Channel(200) %% Max Dealy Spread in us


%% Extracting Data
% Creat Receiver Object
Receiver = IEEE802_11a_Receiver(Effects.TransmitterOutput);%IEEE802_11a_Receiver(TX_Output);
Receiver.EqualizerMode = 1;
RX_Data = Receiver.ReceiveData();

RX_data_bits = dec2bin(RX_Data)-'0';
ByteError = sum(RX_Data ~= data_hex)/LENGTH;
BitError = sum(sum(RX_data_bits ~= data_bits))/(LENGTH*8);


disp("Byte Error: ");disp(ByteError);
disp("Bit Error: ");disp(BitError);
