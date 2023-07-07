T = readtable('plot_data.csv');

%% manhattan metric compare: k = .., steps = 50
vit_err = T.k1_err_vit;
vit_err = vit_err(~isnan(vit_err));
forw_err = T.k1_err_forward;
forw_err = forw_err(~isnan(forw_err));
fb_err = T.k1_err_for_back;
fb_err = fb_err(~isnan(fb_err));
t = 1:1:numel(vit_err);
figure();
hold on, grid on
scatter(t, vit_err, 15,'r' ,'filled');
scatter(t, forw_err, 15,'k' ,'filled');
scatter(t, fb_err, 15, 'b','filled');
yline(mean(vit_err), 'r', 'LineWidth', 1);
yline(mean(forw_err), 'k', 'LineWidth', 1);
yline(mean(fb_err), 'b', 'LineWidth', 1);
legend('Viterbi', 'Forward', 'Forw-back')
xlabel('Iteration'), ylabel('Error')
title('Average error comparison for k=1, steps = 50')

%% hit and miss metric, 50 steps, 30 simulations
vit_hits = T.viterbi_hits_waste;
vit_hits = vit_hits(~isnan(vit_hits));
fb_hits = T.forwback_hits_waste;
fb_hits = fb_hits(~isnan(fb_hits));
forward_hits = T.forward_hits_waste;
forward_hits = forward_hits(~isnan(forward_hits));

total = 50*30;
Y = [vit_hits/total, fb_hits/total, forward_hits/total];
figure()
hold on, grid on
bar(Y);
names = {'Viterbi', 'Forward-backward','Forward'};
text(1:length(Y),Y,num2str(Y'),'vert','bottom','horiz','center'); 
set(gca,'xtick',[1:3],'xticklabel',names)
ylabel('Hit rate')
ylim([0,0.6])
title('Hit rate of the individual algorithms - wasteland')

%% forward probability underflow
forw_probs = T.forward_probabilites;
forwback_probs  = T.forwardbackward_probabilites;
figure()
hold on, grid on
plot(forw_probs, 'LineWidth', 1)
plot(forwback_probs, 'LineWidth', 1)
set(gca, 'YScale', 'log')
legend('Forward', 'Forward-backward')
xlabel('Simulation steps'), ylabel('Highest state probability')
title('Forward(backward) underflow issue')

forw_probs_scaled = T.forward_probabilites_scaled;
forwback_probs_scaled = T.forwardbackward_probabilites_scaled;
figure()
hold on, grid on
plot(forw_probs_scaled, 'LineWidth', 1)
plot(forwback_probs_scaled, 'LineWidth', 1)
legend('Forward', 'Forward-backward')
xlabel('Simulation steps'), ylabel('Highest state probability')
title('Forward(backward) algorithm with scaling')
ylim([0, 1])
xlim([0,300])

%% FB and FB scaling comaparison
matches = T.FB_Fbscaled_matches;
matches = flip(matches);
FB_hits = T.FB_hits;
FB_hits = flip(FB_hits);
FB_scaled_hits = T.FB_scaled_hits;
FB_scaled_hits = flip(FB_scaled_hits);

figure()
hold on, grid on
plot(cumsum(matches), 'LineWidth', 1);
plot(cumsum(FB_hits), 'LineWidth', 1);
plot(cumsum(FB_scaled_hits), '--' ,'LineWidth', 1);
xline(317)
legend('Integrated matches', 'Integrated FB hits', 'Intgreated scaled FB hits')
title('The effect of scaling on Forward(backward) algorithm performance')
xlabel('Iterations')

%% viterbi probability underflow
vit_probs = T.viterbi_underflow;
vit_probs = vit_probs(~isnan(vit_probs));
figure()
hold on, grid on
plot(vit_probs, 'LineWidth', 1)
set(gca, 'YScale', 'log')
xlabel('Simulation steps'), ylabel('Max msg probability')
title('Viterbi algorithm underflow')

%% 
vit_matches = T.vit_guess_match;
vit_hits = T.vit_hits;
vit_log_hits = T.vit_log_hits;

figure()
hold on, grid on
plot(cumsum(vit_matches), 'LineWidth', 1);
plot(cumsum(vit_hits), 'LineWidth', 1);
plot(cumsum(vit_log_hits), '--' ,'LineWidth', 1);
xline(266)
legend('Integrated matches', 'Integrated Vit hits', 'Intgreated logVit hits')
title('The effect logarithmic sum in Viterbi algorithm')
xlabel('Iterations')

