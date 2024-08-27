clc;clear;close all;
tic;

%% Wifi Packet Paramters
LENGTH = 100;      % 1-4095
DataRate_Arr = [6,9,12,18,24,36,48,54];      % 6,9,12,18,24,36,48,54  --6,9,36,54 errors
ModOrder = [2,2,4,4,16,16,64,64];

%% Simulation paramters
MaxSNR = 30;
SNR_Arr = -5:1:MaxSNR;
Iterations = 100;


%% Creating Objects
% Creat Transmiter Object
Transmitter = IEEE802_11a_Transmitter(LENGTH);
% Transmitter.DebugMode = 1;

% Creat Effects Object
Effects = IEEE802_11a_Effects();
% Effects.DebugMode = 1; % Enable Effects Debug Mode
    
% Creat Receiver Object
Receiver = IEEE802_11a_Receiver();%IEEE802_11a_Receiver(TX_Output);
Receiver.PacketDetectionMode = 1;
Receiver.EqualizerMode = 1;
Receiver.CFO_Mode = 1;


Transmitter.DATARATE = DataRate_Arr(6);
SNR_BER = zeros(1,length(SNR_Arr));
SNR_PFR = zeros(1,length(SNR_Arr));
for Snr_index = 1:length(SNR_Arr)
    SNR_Arr(Snr_index)
    BER = []; % Bit Error Rate
    PFR = 0; % Packet Faild Rate
    for i = 1:Iterations
        %% Data Generating
        data_hex = randi(255,LENGTH,1);
        data_bits = dec2bin(data_hex)-'0';
        %% Data Transmittion
        TX_Output = Transmitter.GenerateWaveform(data_hex);
        %% Adding Effects
        Effects.TransmitterOutput = TX_Output;
        Effects.add_CFO(100) %% Ratio of added Carrier offset
        % Effects.add_STO(randi([300,1000])) %% Number of added samples befor the waveform
        Effects.add_Noise(SNR_Arr(Snr_index));
        Effects.add_Channel() %% Max Dealy Spread in us
        %% Receiving Data
        try
            RX_Data = Receiver.ReceiveData(Effects.TransmitterOutput);
    
            RX_data_bits = dec2bin(RX_Data)-'0';
            BitError = sum(sum(RX_data_bits ~= data_bits))/(LENGTH*8);
            BER = [BER BitError];
        catch Error
            % RX_Data = 0;
            %disp('Error Message:')
            %disp(Error.message)
            if (Error.identifier == "Reciver:PacketFaild")
                disp("Packet Faild")
                disp(Error.message)
                % BER = [BER 1];
                PFR = PFR+1;
            else
                disp("Unkown Error")
                disp(Error.message)
                errorName = Error.stack.name;disp(errorName)
                errorLine = Error.stack.line;disp(errorLine)
            end
        end
    end
    
    SNR_BER(Snr_index) = sum(BER)/length(BER);
    SNR_PFR(Snr_index) = PFR/Iterations;
    % ByteError = sum(RX_Data ~= data_hex)/LENGTH;
    
    % disp("Byte Error: ");disp(ByteError);
    % disp("Bit Error: ");disp(BitError);
end

% Ploting BER
figure();
semilogy(SNR_Arr,SNR_BER);
title("BER For DataRate 6")
% Ploting PFR
figure();
plot(SNR_Arr,SNR_PFR);hold on;
stem(SNR_Arr,SNR_PFR);
title("PFR For DataRate 6")