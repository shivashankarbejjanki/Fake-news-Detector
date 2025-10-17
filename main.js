// Main JavaScript file for Fake News Detection App

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Add fade-in animation to main content
    const mainContent = document.querySelector('main');
    if (mainContent) {
        mainContent.classList.add('fade-in');
    }
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Initialize form enhancements
    initializeFormEnhancements();
    
    // Initialize API functionality
    initializeAPI();
    
    // Add smooth scrolling
    initializeSmoothScrolling();
    
    // Initialize character counter
    initializeCharacterCounter();
    
    // Initialize copy to clipboard functionality
    initializeCopyToClipboard();
}

function initializeFormEnhancements() {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    const textarea = document.getElementById('news_text');
    const submitBtn = document.getElementById('analyzeBtn');
    
    if (textarea && submitBtn) {
        // Real-time validation
        textarea.addEventListener('input', function() {
            const text = this.value.trim();
            const isValid = text.length >= 10;
            
            // Update button state
            submitBtn.disabled = !isValid;
            
            // Update textarea styling
            if (text.length > 0) {
                if (isValid) {
                    textarea.classList.remove('border-danger');
                    textarea.classList.add('border-success');
                } else {
                    textarea.classList.remove('border-success');
                    textarea.classList.add('border-danger');
                }
            } else {
                textarea.classList.remove('border-success', 'border-danger');
            }
        });
        
        // Form submission handling
        form.addEventListener('submit', function(e) {
            const text = textarea.value.trim();
            
            if (text.length < 10) {
                e.preventDefault();
                showAlert('Please enter at least 10 characters for analysis.', 'warning');
                textarea.focus();
                return false;
            }
            
            // Show loading state
            showLoadingState(submitBtn);
        });
    }
}

function initializeAPI() {
    // Create API helper functions
    window.FakeNewsAPI = {
        predict: async function(text, model = 'logistic_regression') {
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        model: model
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                return await response.json();
            } catch (error) {
                console.error('API Error:', error);
                throw error;
            }
        },
        
        getModels: async function() {
            try {
                const response = await fetch('/api/models');
                return await response.json();
            } catch (error) {
                console.error('API Error:', error);
                throw error;
            }
        }
    };
}

function initializeSmoothScrolling() {
    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function initializeCharacterCounter() {
    const textarea = document.getElementById('news_text');
    if (!textarea) return;
    
    // Create character counter element
    const counter = document.createElement('div');
    counter.className = 'form-text text-end';
    counter.id = 'char-counter';
    
    // Insert after textarea
    textarea.parentNode.insertBefore(counter, textarea.nextSibling);
    
    // Update counter function
    function updateCounter() {
        const length = textarea.value.length;
        const minLength = 10;
        const maxLength = 10000; // Reasonable limit
        
        counter.innerHTML = `
            <span class="${length < minLength ? 'text-danger' : length > maxLength ? 'text-warning' : 'text-success'}">
                ${length} characters
            </span>
            ${length < minLength ? `(${minLength - length} more needed)` : ''}
            ${length > maxLength ? '(Consider shortening for better performance)' : ''}
        `;
    }
    
    // Initial update and event listener
    updateCounter();
    textarea.addEventListener('input', updateCounter);
}

function initializeCopyToClipboard() {
    // Add copy functionality to code blocks or results
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const target = document.querySelector(this.dataset.target);
            if (target) {
                copyToClipboard(target.textContent);
                showAlert('Copied to clipboard!', 'success');
            }
        });
    });
}

function showLoadingState(button) {
    if (!button) return;
    
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
    button.disabled = true;
    button.classList.add('loading');
    
    // Store original text for potential restoration
    button.dataset.originalText = originalText;
}

function hideLoadingState(button) {
    if (!button) return;
    
    const originalText = button.dataset.originalText;
    if (originalText) {
        button.innerHTML = originalText;
    }
    button.disabled = false;
    button.classList.remove('loading');
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of main content
    const main = document.querySelector('main');
    if (main) {
        main.insertBefore(alert, main.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }
}

function copyToClipboard(text) {
    if (navigator.clipboard) {
        return navigator.clipboard.writeText(text);
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        return Promise.resolve();
    }
}

function animateProgressBars() {
    // Animate progress bars on results page
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach((bar, index) => {
        const width = bar.style.width;
        bar.style.width = '0%';
        
        setTimeout(() => {
            bar.style.transition = 'width 1s ease-in-out';
            bar.style.width = width;
        }, index * 200 + 100);
    });
}

function validateInput(text) {
    // Input validation helper
    const errors = [];
    
    if (!text || text.trim().length === 0) {
        errors.push('Text cannot be empty');
    }
    
    if (text.trim().length < 10) {
        errors.push('Text must be at least 10 characters long');
    }
    
    if (text.length > 10000) {
        errors.push('Text is too long (maximum 10,000 characters)');
    }
    
    // Check for suspicious patterns that might break the model
    const suspiciousPatterns = [
        /^(.)\1{50,}$/, // Repeated characters
        /^[^a-zA-Z]*$/, // No letters at all
    ];
    
    suspiciousPatterns.forEach(pattern => {
        if (pattern.test(text)) {
            errors.push('Text appears to contain invalid patterns');
        }
    });
    
    return {
        isValid: errors.length === 0,
        errors: errors
    };
}

// Utility functions for results page
function highlightPrediction(prediction, confidence) {
    const elements = document.querySelectorAll('.prediction-highlight');
    elements.forEach(el => {
        el.classList.remove('border-success-thick', 'border-danger-thick');
        
        if (prediction === 'Real') {
            el.classList.add('border-success-thick');
        } else {
            el.classList.add('border-danger-thick');
        }
        
        // Add pulse animation for high confidence
        if (confidence > 0.8) {
            el.style.animation = 'pulse 2s infinite';
        }
    });
}

function formatConfidenceScore(confidence) {
    if (confidence === null || confidence === undefined) {
        return 'N/A';
    }
    
    const percentage = (confidence * 100).toFixed(1);
    let label = '';
    
    if (confidence > 0.8) {
        label = 'High';
    } else if (confidence > 0.6) {
        label = 'Moderate';
    } else {
        label = 'Low';
    }
    
    return `${percentage}% (${label})`;
}

// Export functions for use in other scripts
window.FakeNewsApp = {
    showAlert,
    showLoadingState,
    hideLoadingState,
    copyToClipboard,
    animateProgressBars,
    validateInput,
    highlightPrediction,
    formatConfidenceScore
};

// Initialize results page animations if on results page
if (window.location.pathname.includes('predict') || document.querySelector('.progress-bar')) {
    setTimeout(animateProgressBars, 500);
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const form = document.getElementById('predictionForm');
        if (form) {
            form.submit();
        }
    }
    
    // Escape to clear form
    if (e.key === 'Escape') {
        const textarea = document.getElementById('news_text');
        if (textarea && textarea === document.activeElement) {
            textarea.value = '';
            textarea.dispatchEvent(new Event('input'));
        }
    }
});

// Add service worker for offline functionality (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed');
            });
    });
}
