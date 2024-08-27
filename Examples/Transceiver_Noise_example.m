clc;clear;close all;
tic;

%% Wifi Packet Paramters
LENGTH = 100;      % 1-4095                % Transmitter Default Length: 100
DataRate = 36;      % 6,9,12,18,24,36,48,54 % Transmitter Default DataRate: 36

%% Creating Objects
% Creat Transmiter Object
Transmitter = IEEE802_11a_Transmitter(LENGTH,DataRate);
Transmitter.DebugMode = 1; % Enable Transmitter & Receiver Debug Mode

% Creat Effects Object
Effects = IEEE802_11a_Effects();
Effects.DebugMode = 1; % Enable Effects Debug Mode
    
% Creat Receiver Object
Receiver = IEEE802_11a_Receiver();

%% Data Generating
data_hex = randi(255,LENGTH,1);
data_bits = dec2bin(data_hex)-'0';
%% Data Transmittion
TX_Output = Transmitter.GenerateWaveform(data_hex);
%% Adding Effects
Effects.TransmitterOutput = TX_Output;
Effects.add_Noise(5); %% SNR = 5 db

%% Receiving Data
try
    RX_Data = Receiver.ReceiveData(Effects.TransmitterOutput);
catch Error
    RX_Data = 0;
    %disp('Error Message:')
    %disp(Error.message)
    if (Error.identifier == "Reciver:PacketFaild")
        disp("Packet Faild")
        disp(Error.message)
    else
        disp("Unkown Error")
        disp(Error.message)
        errorName = Error.stack.name;disp(errorName)
        errorLine = Error.stack.line;disp(errorLine)
    end
end
%% Calculate Bit Error Rate (BER)
RX_data_bits = dec2bin(RX_Data)-'0';
ByteError = sum(RX_Data ~= data_hex)/LENGTH;
BitError = sum(sum(RX_data_bits ~= data_bits))/(LENGTH*8);

disp("Byte Error: ");disp(ByteError);
disp("Bit Error: ");disp(BitError);

toc;
