clc ; close all; clear; 
% Waveform Generating
LENGTH = 100;
Transmitter = IEEE802_11a_Transmitter(LENGTH, 36);
data_hex = randi(255, LENGTH, 1);
data_bits = dec2bin(data_hex) - '0';
Wifi_Output = Transmitter.GenerateWaveform(data_hex);
Waveform = Wifi_Output.waveform;
TX_waveform= Waveform ; 
STS = TX_waveform(1:16);
STS_Waveform = [] ; 
% ========================================================================================================
% ==================================== Sync Parameters ===================================================
FFT_Size = 64;   
SNR_dB = 10:10 ; 
cross_threshold = 0.2 ; %[0.5 , 0.6 , 0.7 , 0.72 , 0.73 ] ; 
Power_threshold = 0.001 ; %[0.2,0.25 , 0.3 ,0.35 , 0.4]; -->0.25 is good for all effect and total range of SNR 
resolution_samples = 0 ; 
resolution_samples2 = 0 ; 

% =================================== Adding Effects =======================================================
%  1) CFO ===========================================
Carrier_freq = 5.8e9;  % Carrier frequency of Wi-Fi 802.11a (5.8 GHz)
Tolerance = 20e-6;     % ppm for Tx or RX
MAX_Offset = 2 * (Tolerance) * (Carrier_freq);  % Assuming TX and RX cause Carrier Offset together
Sampling_Freq = 20e6; 
SubCarrier_Spacing = Sampling_Freq / FFT_Size; 
Ratio = 1; 
delta_f = Ratio * MAX_Offset; % Frequency offset in Hz
epsilon = delta_f / (SubCarrier_Spacing * FFT_Size); % CFO in terms of sampling frequency or SubCarrier Spacing 
%  Applying CFO to entire packet
n = (0:length(Waveform) - 1);  
phase_shift = exp(1j * n * 2 * pi * epsilon);
Waveform = Waveform .* phase_shift.';
CFO_Error = 0 ; 

%% Channel parameters =============================
paths = 4;
path_gains = (1/sqrt(2)) * (randn(1, paths) + 1i * randn(1, paths)); 
%% Adding Channel =================================  
%  Waveform = conv(Waveform, path_gains, 'same'); 

%% Adding Zeros =================================== 
added_Samples = randi([100, 200])  
Waveform = [zeros(added_Samples, 1); Waveform]; 

%% ================================== Packet Detection =======================================================
for snr = SNR_dB 
    snr 
    % Adding Noise======================= %   
        Waveform = Adding_Noise_(Waveform, snr);
        %% Power Trigger Block 
        Power_out = zeros(1, length(Waveform));
        max_power = 1  ;%abs(Waveform).^2;
        for index = 1:length(Power_out)
            Power_out(index) = Waveform(index) * conj(Waveform( index));
        end
        Power_norm = Power_out / max(max_power);
        % Plot Power
        figure;
        plot(Power_norm);
        title(['Power Normalized for SNR = ', num2str(snr), ' dB']);
        power_peaks       = find(Power_norm >= Power_threshold); 
        signal_start      = power_peaks(1)     
            power_start_Error = signal_start - (added_Samples+1)  % Calculated not estimated
            %% Waveform Buffer 
            Waveform_buffer   = Waveform(signal_start:signal_start+160-1);

            %% First corr
            corr_10_output = myXcorr(STS,Waveform_buffer);
            % Maximum Normalized Correlation (MNC)
            norm_factor = abs(sum(Waveform_buffer .* conj(Waveform_buffer))); % The power of STS
            % Normalize the cross-correlation output
            corr_10_output = abs(corr_10_output).^2 / (norm_factor^2); 
            corr_10_output_norm = corr_10_output / max(corr_10_output); 
              
                    % Plotting
                    figure;
                    plot(abs(corr_10_output_norm));
                    title(['Cross-correlation for SNR = ' , num2str(snr), ' dB']);

            % first est
           [ Peaks_Window_1,Values ] = find(corr_10_output_norm >= cross_threshold); 
            
            %% Peak search 
                 counter =0 ; 
                 buffer =[] ; 
                    if Peaks_Window_1 >= 5
                        i = 1;
                        while ( i < length(Peaks_Window_1))
                            if (i+1 <= length(Peaks_Window_1))
                                distance =  Peaks_Window_1(i+1) - Peaks_Window_1(i); 
                                if distance >= 14
                                    counter = counter + 1;
                                    buffer= [buffer , Peaks_Window_1(i) ];
                                    i=i+1 ;     
                                        else
                                           if (Values(i) > Values(i+1)) 
                                                buffer= [buffer ,Peaks_Window_1(i) , Peaks_Window_1(i+2:end) ];
                                                i=1 ; 
                                              else
                                                buffer= [buffer ,Peaks_Window_1(i+1) ] ;
                                                i = i + 1;  
                                           end 
                                end 
                                
                            else
                                break;  
                            end
                        end
                    end

                    
            STS_index = buffer(1); 
            Xcorr_Error1 = STS_index - 16; % -2  // 2 
            % First Correction
                    if(Xcorr_Error1<0)
                             STS_9 = Waveform_buffer(17+Xcorr_Error1:end+Xcorr_Error1);
                          else
                             STS_9 = [Waveform_buffer(17+Xcorr_Error1:end);zeros(Xcorr_Error1,1)];   
                    end
                    
                    
            %% Second corr 
            corr_9_output    = myXcorr(STS,STS_9);
            % Maximum Normalized Correlation (MNC)
            norm_factor = abs(sum(Waveform_buffer .* conj(Waveform_buffer))); % The power of STS
            % Normalize the cross-correlation output
            corr_9_output = abs(corr_9_output).^2 / (norm_factor^2); 
            corr_9_output_norm = corr_9_output / max(corr_9_output); 
                        % Plotting
                        figure;
                        plot(abs(corr_9_output_norm));
                        title(['Cross-correlation for SNR = ' , num2str(snr), ' dB']);     
            % Second est   
            [Peaks_Window_2 ]  = find(corr_9_output_norm >= cross_threshold);
            Second_STS_index = Peaks_Window_2(1);
            Xcorr_Error2  = Second_STS_index - 16 
            short_seq = STS_9(Second_STS_index+1:end);
            Coarse_Esti_Epsilon = Coarse_CFO_estimation(short_seq(1:32)) 
end
 

 %% Noise

function [Rx_waveform_Noise] = Adding_Noise_(TX_Waveform,SNR)
SNR_linear = 10.^(SNR/10);
% Signal power
Waveform_power = mean(TX_Waveform.^2);
Noise_power = (Waveform_power/SNR_linear);

% Addition of noise
Noise = sqrt(Noise_power / 2)*(randn(length(TX_Waveform), 1) + 1i * randn(length(TX_Waveform), 1));

Rx_waveform_Noise = TX_Waveform + Noise;

 end

 
function [ correlation] = myXcorr(STS,Waveform )
    len_STS = length(STS);
    len_Waveform = length(Waveform);
    % correlation = zeros(1, len_STS + len_Waveform - 1);
    correlation = conv(Waveform, flip(conj(STS)));
    % Maximum Normalized Correlation (MNC)
    norm_factor = abs(sum(Waveform .* conj(Waveform))); % The power of STS
    % Normalize the cross-correlation output
    correlation = abs(correlation).^2 / (norm_factor^2); 
    correlation = correlation / max(correlation); 
    
      
end 


 function [Coarse_Esti_Epsilon] = Coarse_CFO_estimation(RX_Waveform_Coarse) 

    %% Coarse CFO in freq domain
    % estimation using Short preamble Symbols (STS)
    Y1_STS_Sample_freq = fft(RX_Waveform_Coarse(1:16), 16); % Y1
    Y2_STS_Sample_freq = fft(RX_Waveform_Coarse(17:32), 16); % Y2 (delayed)
    var1 = sum(imag(Y2_STS_Sample_freq .* conj(Y1_STS_Sample_freq)));
    var2 = sum(real(Y2_STS_Sample_freq .* conj(Y1_STS_Sample_freq)));
    Coarse_Esti_Epsilon = atan2(var1, var2) / (2 * pi*16);

   
end

 
