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
                title("Waveform Before Noise")
                subplot(3,1,2)
                plot(1:length(Noise),abs(Noise))
                title("Additive white Gaussian Noise")
                subplot(3,1,3)
                plot(1:length(waveformNoise),abs(waveformNoise))
                title("Waveform After Noise")
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
                title("Waveform Before STO")
                subplot(2,1,2);
                plot(1:length(waveformSTO),abs(waveformSTO))
                title("Waveform After STO")
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
                disp('CFO Efffect: ');disp(epsilon);
                ActiveSC = IEEE802_11a_Effects.ActiveSC_Extract(obj.TransmitterOutput.waveform);
                figure("Name","Befor CFO Effect");
                plot(ActiveSC,'bx');
                title("Constellation Before CFO")
                ActiveSC = IEEE802_11a_Effects.ActiveSC_Extract(waveformCFO);
                figure("Name","After CFO Effect");
                plot(ActiveSC,'x');
                title("Constellation After CFO")
            end

            obj.TransmitterOutput.waveform = waveformCFO;
        end

        function add_Channel(obj,MaxDelaySpread,Channel_Type,K_Factor)
            %% MaxDelaySpread must be one of thies values 
            %% [ 50 , 100 , 150 , 200 ]
            %% depend on the channel model

            %% K_Factor is 0 By defult for Rayleigh Channel
            %% K_Factor Must be bigger than 0 for Racian Channel
            % MaxDelaySpread = 200;

            switch nargin
                case 1
                    MaxDelaySpread = 200;
                    Channel_Type = "Rayleigh";
                    K_Factor = 0;
                case 2
                    Channel_Type = "Rayleigh";
                    K_Factor = 0;
                case 3
                    if isequal(Channel_Type,"Rayleigh")
                        K_Factor = 0;
                    elseif isequal(Channel_Type,"Racian")
                        K_Factor = 1;
                    end
            end

            %% Channel Creation
            paths = MaxDelaySpread/50; % Number of pathes
            if isequal(Channel_Type,"Rayleigh")
                path_gains = (1/sqrt(2))*(randn(1, paths) + 1i * randn(1, paths));
            elseif isequal(Channel_Type,"Racian")
                % path_gains = ??
            end
            
            %% Add Channel Effect
            waveformChannel = conv(obj.TransmitterOutput.waveform,path_gains,'same');
            
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
                title("Waveform Before Channel")
                subplot(2,1,2)
                plot(1:length(waveformChannel),abs(waveformChannel))
                title("Waveform After Channel")
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