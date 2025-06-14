function plot_final_comparison()
    % --- 1. Hardcode the results from both models ---
    
    % Use a cell array for distances to handle string labels
    distances_str = {'5 ft', '10 ft', '20 ft', '30 ft', '40 ft'};
    
    % Results from the final, successful "Conditional Meta-Model"
    meta_model_f1 = [1.00, 1.00, 1.00, 0.92, 0.99];
    
    % Results from the original "Autoencoder Only" model (the baseline)
    autoencoder_only_f1 = [0.41, 0.08, 0.19, 0.92, 0.00];

    % --- 2. Create the Plot ---
    
    % Create a new figure window and set its size
    figure('Position', [100, 100, 1000, 700]);
    hold on; % Hold on to plot multiple lines on the same axes

    % Define colors
    meta_color = [0.85, 0.16, 0.11]; % A strong red: [R, G, B]
    auto_color = [0.12, 0.47, 0.71]; % A standard blue: [R, G, B]
    highlight_color = [0.95, 0.6, 0.1]; % A bright orange
    
    % ### NEW: Define a better green color ###
    dark_green = [0, 0.6, 0.2];
    
    % Plot the "Autoencoder Only" performance line (the baseline)
    plot(1:length(distances_str), autoencoder_only_f1, ...
        '--s', ... % Dashed line with square markers
        'Color', auto_color, ...
        'LineWidth', 3, ... 
        'MarkerSize', 12, ...
        'MarkerFaceColor', auto_color, ...
        'DisplayName', 'Baseline: Autoencoder Only');

    % Plot the final "Conditional Meta-Model" performance line
    plot(1:length(distances_str), meta_model_f1, ...
        '-o', ... % Solid line with circle markers
        'Color', meta_color, ...
        'LineWidth', 4, ...
        'MarkerSize', 14, ...
        'MarkerFaceColor', meta_color, ...
        'DisplayName', 'Meta-Model (Classifier)');

    % --- 3. Highlight the 30 ft Autoencoder Point ---
    
    % Find the index for '30 ft'
    idx_30ft = find(strcmp(distances_str, '30 ft'));
    
    % Plot a large, distinct diamond marker for the 30 ft point
    plot(idx_30ft, meta_model_f1(idx_30ft), ...
        'd', ... % 'd' is a diamond marker
        'MarkerSize', 22, ... % Make it large to stand out
        'MarkerEdgeColor', 'black', ...
        'MarkerFaceColor', highlight_color, ...
        'LineWidth', 2, ...
        'DisplayName', 'Meta-Model (Autoencoder)');

    % --- 4. Customize for Publication Quality ---
    
    % Set all text properties to be large and bold
    ax = gca; % Get current axes
    ax.FontSize = 16;
    ax.FontWeight = 'bold';
    
    % Set titles and labels
    %title('Final Meta-Model Outperforms Baseline Autoencoder', 'FontSize', 22, 'FontWeight', 'bold');
    %subtitle('Hybrid architecture leverages specialized models for optimal performance', 'FontSize', 16);
    
    xlabel('UE Distance from Base Station (ft)', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('Anomaly Detection F1-Score', 'FontSize', 18, 'FontWeight', 'bold');
    
    % Customize ticks
    xticks(1:length(distances_str));
    xticklabels(distances_str);
    ylim([-0.05, 1.1]);
    grid on;
    
    % ### UPDATED: Use the better green and hide the legend entry ###
    yline(0.9, '--', '90% Threshold', 'LineWidth', 3, 'FontSize', 14, ...
        'Color', dark_green, 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');

    % Add text labels for data points for precision
    for i = 1:length(meta_model_f1)
        text(i, meta_model_f1(i) + 0.05, sprintf('%.2f', meta_model_f1(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'Color', meta_color);
    end
    for i = 1:length(autoencoder_only_f1)
        text(i, autoencoder_only_f1(i) - 0.05, sprintf('%.2f', autoencoder_only_f1(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'Color', auto_color);
    end
    
    % Customize the legend
    lgd = legend('show');
    lgd.Location = 'South';
    lgd.Orientation = 'vertical';
    lgd.FontSize = 12;
    
    hold off;
    
    % --- 5. Save the figure ---
    output_filename = 'model_comparison_matlab.png';
    saveas(gcf, output_filename);
    
    fprintf("Plot successfully saved to '%s'\n", output_filename);

end