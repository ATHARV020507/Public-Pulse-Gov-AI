document.addEventListener('DOMContentLoaded', () => {
    
    // --- DOM Elements ---
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('text-input');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultContainer = document.getElementById('result-container');
    
    const sentimentSpan = document.getElementById('sentiment-result');
    const confidenceSpan = document.getElementById('confidence-result');
    const topicsDiv = document.getElementById('topics-result');
    const summaryP = document.getElementById('summary-result');
    const sourceSpan = document.getElementById('analysis-source-result');
    const timeSpan = document.getElementById('analysis-time-result');

    const tableBody = document.getElementById('comment-table-body');
    const sentimentFilter = document.getElementById('sentiment-filter');
    const wordCloudImg = document.getElementById('word-cloud-img');

    // Bulk Upload Elements
    const modeSingle = document.getElementById('mode-single');
    const modeBulk = document.getElementById('mode-bulk');
    const viewSingle = document.getElementById('view-single');
    const viewBulk = document.getElementById('view-bulk');
    const fileInput = document.getElementById('file-upload');
    const uploadBtn = document.getElementById('upload-btn');
    const singleResultContent = document.getElementById('single-result-content');
    const bulkResultContent = document.getElementById('bulk-result-content');
    const bulkMsg = document.getElementById('bulk-msg');

    // Modal Elements
    const modal = document.getElementById('welcome-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');

    // Settings Elements
    const darkModeToggle = document.getElementById('dark-mode-toggle');

    let allComments = []; 
    let sentimentChartInstance = null;
    let topicChartInstance = null;
    let refreshInterval = null;

    // --- WELCOME MODAL LOGIC ---
    setTimeout(() => {
        if(modal) modal.classList.add('show');
    }, 500);
    if(closeModalBtn) {
        closeModalBtn.addEventListener('click', () => {
            modal.classList.remove('show');
        });
    }

    // --- 1. DARK MODE LOGIC ---
    if (localStorage.getItem('darkMode') === 'enabled') {
        document.body.classList.add('dark-mode');
        if(darkModeToggle) darkModeToggle.checked = true;
    }
    if (darkModeToggle) {
        darkModeToggle.addEventListener('change', () => {
            if (darkModeToggle.checked) {
                document.body.classList.add('dark-mode');
                localStorage.setItem('darkMode', 'enabled');
            } else {
                document.body.classList.remove('dark-mode');
                localStorage.setItem('darkMode', 'disabled');
            }
        });
    }

    // --- 2. CHART & STATS LOGIC ---
    function updateCharts(comments) {
        const sentimentCounts = { 'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Urgent': 0 };
        const topicCounts = {};
        
        let urgentCount = 0;
        let negativeCount = 0;

        comments.forEach(c => {
            let s = c.sentiment;
            if (s === 'Urgent Negative') { s = 'Urgent'; urgentCount++; }
            if (s === 'Negative' || s === 'Urgent') {
                negativeCount++;
                c.topics.forEach(t => { topicCounts[t] = (topicCounts[t] || 0) + 1; });
            }
            if (sentimentCounts[s] !== undefined) sentimentCounts[s]++;
        });

        // UPDATE SUMMARY CARDS
        const totalEl = document.getElementById('total-comments');
        const sentimentEl = document.getElementById('overall-sentiment');
        const urgentEl = document.getElementById('urgent-alerts');

        if (totalEl) totalEl.textContent = comments.length.toLocaleString();
        if (urgentEl) urgentEl.textContent = urgentCount;
        if (sentimentEl && comments.length > 0) {
            const totalNeg = sentimentCounts['Negative'] + sentimentCounts['Urgent'];
            const negPercent = Math.round((totalNeg / comments.length) * 100);
            sentimentEl.textContent = `${negPercent}% Negative`;
        }

        // DRAW CHARTS
        const chart1Canvas = document.getElementById('sentimentChart');
        if (chart1Canvas) {
            const ctx1 = chart1Canvas.getContext('2d');
            if (sentimentChartInstance) sentimentChartInstance.destroy();
            sentimentChartInstance = new Chart(ctx1, {
                type: 'doughnut',
                data: {
                    labels: ['Positive', 'Negative', 'Neutral', 'Urgent'],
                    datasets: [{
                        data: [sentimentCounts['Positive'], sentimentCounts['Negative'], sentimentCounts['Neutral'], sentimentCounts['Urgent']],
                        backgroundColor: ['#28a745', '#ffc107', '#6c757d', '#dc3545'],
                        borderWidth: 0
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { boxWidth: 12 } } } }
            });
        }

        const chart2Canvas = document.getElementById('topicChart');
        if (chart2Canvas) {
            const sortedTopics = Object.entries(topicCounts).sort((a, b) => b[1] - a[1]).slice(0, 5);
            const ctx2 = chart2Canvas.getContext('2d');
            if (topicChartInstance) topicChartInstance.destroy();
            topicChartInstance = new Chart(ctx2, {
                type: 'bar',
                data: {
                    labels: sortedTopics.map(i => i[0]),
                    datasets: [{
                        label: 'Frequency',
                        data: sortedTopics.map(i => i[1]),
                        backgroundColor: '#007bff',
                        borderRadius: 4
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } } }
            });
        }
    }

    // --- 3. REFRESH WORD CLOUD ---
    function refreshWordCloud(sentiment) {
        const timestamp = new Date().getTime();
        if(wordCloudImg) {
            wordCloudImg.src = `/wordcloud.png?sentiment=${sentiment}&t=${timestamp}`;
        }
    }

    // --- 4. RENDER TABLE (FIXED FOR URGENT LABELS) ---
    function renderTable(commentsToRender) {
        if(!tableBody) return;
        tableBody.innerHTML = ''; 
        
        if (commentsToRender.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:#999; padding:20px;">No comments found.</td></tr>';
            return;
        }
        
        commentsToRender.forEach(comment => {
            // 1. Format Topics
            let topicsHTML = comment.topics.map(topic => 
                `<span class="topic-tag">${topic}</span>`
            ).join(' ');

            // 2. Format Sentiment (THE FIX)
            let sentimentClass = '';
            let sentimentText = '';

            // If the backend says "Urgent Negative", force the RED 'urgent' class
            if (comment.sentiment === 'Urgent Negative') {
                sentimentClass = 'urgent';      // This maps to .sentiment-tag.urgent (RED)
                sentimentText = 'URGENT';       // Make text uppercase for impact
            } 
            // Otherwise, just handle Positive/Negative/Neutral
            else {
                sentimentClass = comment.sentiment.toLowerCase();
                sentimentText = comment.sentiment;
            }
            
            let sentimentHTML = `<span class="sentiment-tag ${sentimentClass}">${sentimentText}</span>`;

            // 3. Create Row
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${comment.id}</strong></td>
                <td>${sentimentHTML}</td>
                <td>${topicsHTML}</td>
                <td>${comment.summary}</td>
            `;
            tableBody.appendChild(row);
        });
    }
    // --- 5. LOAD ALL COMMENTS ---
    function loadAllComments() {
        fetch('/get_all_comments')
            .then(response => response.json())
            .then(data => {
                allComments = data; 
                if (sentimentFilter) {
                    const currentFilter = sentimentFilter.value;
                    const filtered = currentFilter === 'all' ? allComments : allComments.filter(c => c.sentiment.toLowerCase().replace(' ', '-') === currentFilter);
                    renderTable(filtered);
                }
                updateCharts(allComments);
            })
            .catch(error => console.error('Error:', error));
    }

    // --- 6. AUTO REFRESH ---
    function startAutoRefresh() {
        if (refreshInterval) clearInterval(refreshInterval);
        refreshInterval = setInterval(() => {
            loadAllComments(); 
            if(sentimentFilter) refreshWordCloud(sentimentFilter.value);
        }, 10000); 
    }

    // --- 7. UI TOGGLES ---
    if(modeSingle && modeBulk) {
        modeSingle.addEventListener('click', () => {
            viewSingle.classList.remove('hidden');
            viewBulk.classList.add('hidden');
            modeSingle.className = 'mode-btn active';
            modeBulk.className = 'mode-btn inactive';
            resultContainer.classList.add('hidden');
        });
        modeBulk.addEventListener('click', () => {
            viewSingle.classList.add('hidden');
            viewBulk.classList.remove('hidden');
            modeBulk.className = 'mode-btn active';
            modeSingle.className = 'mode-btn inactive';
            resultContainer.classList.add('hidden');
        });
    }

    // --- 8. ANALYZE SINGLE ---
    if(analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            const textToAnalyze = textInput.value;
            if (textToAnalyze.trim() === '') { alert('Enter text first.'); return; }
            resultContainer.classList.add('hidden');
            if(singleResultContent) singleResultContent.classList.remove('hidden');
            if(bulkResultContent) bulkResultContent.classList.add('hidden');
            loadingSpinner.classList.remove('hidden');
            analyzeBtn.disabled = true;
            
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: textToAnalyze }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                sentimentSpan.textContent = data.sentiment;
                confidenceSpan.textContent = data.confidence;
                summaryP.textContent = data.summary;
                sourceSpan.textContent = `Source: ${data.analysis_source}`;
                timeSpan.textContent = `Time: ${data.analysis_time}`;
                topicsDiv.innerHTML = data.key_topics.map(t => `<span class="topic-tag">${t}</span>`).join(' ');
                loadAllComments();
                if(sentimentFilter) refreshWordCloud(sentimentFilter.value);
                loadingSpinner.classList.add('hidden');
                resultContainer.classList.remove('hidden');
                analyzeBtn.disabled = false;
            })
            .catch(error => {
                alert(error.message);
                loadingSpinner.classList.add('hidden');
                analyzeBtn.disabled = false;
            });
        });
    }

    // --- 9. BULK UPLOAD ---
    if(uploadBtn) {
        uploadBtn.addEventListener('click', () => {
            if (fileInput.files.length === 0) { alert("Select a file first!"); return; }
            loadingSpinner.classList.remove('hidden');
            resultContainer.classList.add('hidden');
            uploadBtn.disabled = true;
            uploadBtn.textContent = "Processing...";
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            fetch('/upload_csv', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                loadingSpinner.classList.add('hidden');
                uploadBtn.disabled = false;
                uploadBtn.textContent = "Upload & Process";
                if (data.error) { alert("Error: " + data.error); } else {
                    resultContainer.classList.remove('hidden');
                    if(singleResultContent) singleResultContent.classList.add('hidden');
                    if(bulkResultContent) bulkResultContent.classList.remove('hidden');
                    bulkMsg.textContent = data.message;
                    loadAllComments();
                    refreshWordCloud('all');
                }
            })
            .catch(error => {
                alert("Server Error.");
                loadingSpinner.classList.add('hidden');
                uploadBtn.disabled = false;
            });
        });
    }

    // --- 10. FILTER CHANGE ---
    if(sentimentFilter) {
        sentimentFilter.addEventListener('change', () => {
            const selectedValue = sentimentFilter.value;
            const filtered = selectedValue === 'all' ? allComments : allComments.filter(c => c.sentiment.toLowerCase().replace(' ', '-') === selectedValue);
            renderTable(filtered);
            refreshWordCloud(selectedValue);
        });
    }

    // --- 11. INIT ---
    loadAllComments(); 
    startAutoRefresh();
});