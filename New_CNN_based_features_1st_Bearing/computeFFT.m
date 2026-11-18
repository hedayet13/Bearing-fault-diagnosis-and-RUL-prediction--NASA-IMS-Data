%% Local Function 2: Compute FFT Without Normalization
%% ============================================================

function P1 = computeFFT(rawSignal, L)

Y = fft(rawSignal);

P2 = abs(Y / L);

P1 = P2(1:L/2+1);

P1(2:end-1) = 2 * P1(2:end-1);

P1 = single(P1(:));

end
