classdef IEEE802_11a_Effects < handle
    properties(Access = public)
        %% Waveform
        % Waveform  % Transmitter output waveform
        TransmitterOutput
        DebugMode % Enable the channel effects ploting

    end
    
    properties(Access = private)
        % Waveform    % Transmitter output waveform
    end

    methods(Access = public)

        function obj = IEEE802_11a_Effects(Transmitter_Output,DebugMode)
            if nargin == 1
                obj.TransmitterOutput = Transmitter_Output;
                obj.DebugMode = 0;
            elseif nargin == 2
                obj.TransmitterOutput = Transmitter_Output;
                obj.DebugMode = DebugMode;
            end
        end

        function add_Noise(obj,SNR)
            %% Signal to Noise Ratio (SNR) With db
            if nargin == 1
                SNR = 30;
            end
            
            SNR_linear = 10.^(SNR/10);
            % Signal power
            Waveform_power = mean((obj.TransmitterOutput.waveform).^2);
            Noise_power = (Waveform_power/SNR_linear);
            
            % Addition of noise
            Noise = sqrt(Noise_power / 2)*(randn(length(obj.TransmitterOutput.waveform), 1) ...
                                       + 1i * randn(length(obj.TransmitterOutput.waveform), 1));
            waveformNoise = obj.TransmitterOutput.waveform + Noise;
            
            %% Channel Ploting
            if (obj.DebugMode)
                figure("Name","Noise Effect")
                title("Noise Effect")
                subplot(3,1,1)
                plot(1:length(obj.TransmitterOutput.waveform),abs(obj.TransmitterOutput.waveform))
                title("Waveform Before Adding Noise")
                subplot(3,1,2)
                plot(1:length(Noise),abs(Noise))
                title("Additive white Gaussian Noise")
                subplot(3,1,3)
                plot(1:length(waveformNoise),abs(waveformNoise))
                title("Waveform After Adding Noise")
            end
            
            %% Saving waveform with noise effect
            obj.TransmitterOutput.waveform = waveformNoise;
        end

        function add_STO(obj,SamplesNumber)
            %% Samples Number is the number of time doamin sampls wanted to add
            waveformSTO = [zeros(SamplesNumber,1) ; obj.TransmitterOutput.waveform];

            if(obj.DebugMode)
                disp("Numper Of Added Samples: ");disp(SamplesNumber);
                figure("Name","STO Effect");
                subplot(2,1,1);
                plot(1:length(obj.TransmitterOutput.waveform),abs(obj.TransmitterOutput.waveform))
                title("Waveform Before Adding STO")
                subplot(2,1,2);
                plot(1:length(waveformSTO),abs(waveformSTO))
                title("Waveform After Adding STO")
            end
            obj.TransmitterOutput.waveform = waveformSTO;
        end

        function add_CFO(obj,Ratio)
            %% Ratio [0:100] CFO Ratio Effect
            %% CFO Paramters  
            Fs = 20e6;
            Carrier_freq = 5.8e9;  % Carrier frequency of Wi-Fi 802.11a (5.8 GHz)
            Tolerance = 20e-6;     % ppm for Tx or RX
            MAX_Offset = 2 * (Tolerance) * (Carrier_freq);  % Assuming TX and RX cause Carrier Offset together
            % Ratio = 100; %% 
            delta_f = (Ratio/100) * MAX_Offset; % Frequency offset in Hz
            epsilon = delta_f / Fs; % CFO in terms of sampling frequency Or SubCarrier Spacing 
            %% Adding Effect
            n = (0:length(obj.TransmitterOutput.waveform)-1);  
            phase_shift = exp(1j * n * 2 * pi * epsilon).';
            waveformCFO = obj.TransmitterOutput.waveform .* phase_shift;

            if(obj.DebugMode)
                disp('CFO Efffect (Hz): ');disp(epsilon*Fs);
                ActiveSC = IEEE802_11a_Effects.ActiveSC_Extract(obj.TransmitterOutput.waveform);
                figure("Name","Befor CFO Effect");
                plot(ActiveSC,'bx');
                title("Constellation Before Adding CFO")
                ActiveSC = IEEE802_11a_Effects.ActiveSC_Extract(waveformCFO);
                figure("Name","After CFO Effect");
                plot(ActiveSC,'x');
                title("Constellation After Adding CFO")
            end

            obj.TransmitterOutput.waveform = waveformCFO;
        end

        function add_Channel(obj,Channel_Type,K_Factor_dB)

            %% K_Factor is 0 By defult for Rayleigh Channel
            %% K_Factor Must be bigger than 0 for Racian Channel

            Ts=1/20e6;              % sampling Rate
            sigma_t=25e-9;          % RMS delay spread
            lmax = ceil(10*sigma_t/Ts);  % Maximum number of paths
            l=0:lmax-1; 
            sigma02=(1-exp(-Ts/sigma_t))/(1-exp(-(lmax+1)*Ts/sigma_t)); % Power of the first tap
            PDP = sigma02*exp(-l*Ts/sigma_t); %% power delay profile

            switch nargin
                case 1
                    Channel_Type = "Rayleigh";
                    K_Factor_dB = 0;
                case 2
                    if isequal(Channel_Type,"Rayleigh")
                        K_Factor_dB = 0;
                    elseif isequal(Channel_Type,"Racian")
                        K_Factor_dB = 1;
                    end
               case 3
                   if isequal(Channel_Type,"Rayleigh")
                        K_Factor_dB = 0;
                   end
            end

            %% Channel Creation

            h =((randn(1,lmax)+1i*randn(1,lmax))/sqrt(2) )  .* sqrt(PDP);  %% channel impulse 

            if isequal(Channel_Type,"Rayleigh")
                path_gains = (1/sqrt(2))*(randn(1, lmax) + 1i * randn(1, lmax)) .* sqrt(PDP);
            elseif isequal(Channel_Type,"Racian")
                K = 10^(K_Factor_dB/10);
                path_gains = sqrt(K/(K+1))+sqrt(1/(K+1)) ...
                             * (1/sqrt(2))*(randn(1, lmax) + 1i * randn(1, lmax))...
                             .* sqrt(PDP);
            end
            
            %% Add Channel Effect
            waveformChannel = conv(obj.TransmitterOutput.waveform,path_gains,'same');
            
            % %% average power of the channel
            % avg_pow_h=zeros(1,length(PDP));
            % for k = 1:length(PDP)
            % avg_pow_h(k)= mean(h(:,k).*conj(h(:,k)));
            % end

            %% Channel + Effect Ploting 
            if (obj.DebugMode)
                figure("Name","Channel Visualization")
                subplot(2,1,1)
                stem(abs(path_gains));
                title("Impulse Response")
                subplot(2,1,2)
                plot(abs(fft(path_gains,64)))
                title("Frequancy Response")

                figure("Name","Channel Effect")
                subplot(2,1,1)
                plot(1:length(obj.TransmitterOutput.waveform),abs(obj.TransmitterOutput.waveform))
                title("Waveform Before Adding Channel")
                subplot(2,1,2)
                plot(1:length(waveformChannel),abs(waveformChannel))
                title("Waveform After Adding Channel")

                % figure;
                % stem([0:length(PDP)-1],PDP,'ko');   %% plotting power delay profile
                % title("Power delay profile ( Exponential distribution)");
                % xlabel("Channel tap index");
                % ylabel("power");
                
                % figure;
                % stem([0:length(h)-1],avg_pow_h,'ro'); % plotting the channel taps vs power
                % xlabel("Channel tap index");
                % ylabel("power");
                % title("Channel Impulse Response");
            end
            %% Save channel Effect
            obj.TransmitterOutput.waveform = waveformChannel;

        end

    end

    methods(Access = private,Static)
        function ActiveSC = ActiveSC_Extract(waveform)
            waveform_reshaped = reshape(waveform,80,[]);
            waveform_reshaped(:,1:5) = [];
            waveform_reshaped(1:16,:) = [];
            % subcarriers = waveform_reshaped(16+1:end,:);
            SC_FreqDoamin = circshift(fft(waveform_reshaped),32);
            ActiveSC = IEEE802_11a_Effects.Gaurd_Remover(SC_FreqDoamin);
        end

        function activeSubCarriers = Gaurd_Remover(OFDM_Symbol_FreqDomain)
        activeSubCarriers = OFDM_Symbol_FreqDomain(7:59,:);
        activeSubCarriers = [activeSubCarriers(1:26,:);  % 23
                            activeSubCarriers(28:end,:)]; 
        end
        
    end
end