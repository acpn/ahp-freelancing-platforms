# Talent Allocation Optimizer

## Overview
This Python-based project implements a talent allocation optimizer to match freelancers with job demands. It leverages the **Analytic Hierarchy Process (AHP)** for multi-criteria decision-making, enabling precise selection of the best-fit talent for each job. By integrating tools like **SerpAPI** to fetch job data and **ChatGPT API** for processing free text, the system provides a robust solution for optimizing human resource allocation.

The project was developed as part of an MBA thesis focused on optimizing resource allocation using **Operational Research** and **Integer Linear Programming** techniques.

---

## Key Features
- **AHP-Based Decision-Making**: 
  - Evaluates talents against multiple criteria such as hourly rate, skills, experience, and English proficiency.
  - Uses pairwise comparisons to determine the best-fit talent for each job.
  
- **Dynamic Talent Matching**: 
  - Matches jobs and talents based on customizable scenarios (low demand , high demand, medium demand).
  - Supports both AHP and NoN-AHP allocation methods for performance comparison.

- **Profit Analysis**: 
  - Calculates the profitability of allocations based on talent and job rates.
  - Provides insights into shortages and surplus talents.

- **Visualization**: 
  - Generates comparison plots for profit, talent shortages, and surplus across different scenarios.

---

## Installation
### Prerequisites
- Python 3.8+
- Poetry (for dependency management)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/talent-allocation-optimizer.git
   
   cd talent-allocation-optimizer

2. Install dependencies

  ```bash
   poetry install
  ```

3. Execute the project
  ```bash
   poetry run exec
  ```

## Overview
This Python-based project implements a talent allocation optimizer to match freelancers with job demands. It leverages the **Analytic Hierarchy Process (AHP)** for multi-criteria decision-making, enabling precise selection of the best-fit talent for each job. By integrating tools like **SerpAPI** to fetch job data and **ChatGPT API** for processing free text, the system provides a robust solution for optimizing human resource allocation.

The project was developed as part of an MBA thesis focused on optimizing resource allocation using **Operational Research** and **Integer Linear Programming** techniques.

---

## Key Features
- **AHP-Based Decision-Making**: 
  - Evaluates talents against multiple criteria such as hourly rate, skills, experience, and English proficiency.
  - Uses pairwise comparisons to determine the best-fit talent for each job.
  
- **Dynamic Talent Matching**: 
  - Matches jobs and talents based on customizable scenarios (e.g., high demand, medium demand).
  - Supports both AHP and non-AHP allocation methods for performance comparison.

- **Profit Analysis**: 
  - Calculates the profitability of allocations based on talent and job rates.
  - Provides insights into shortages and surplus talents.

- **Visualization**: 
  - Generates comparison plots for profit, talent shortages, and surplus across different scenarios.

---

## Results
### Summary of Findings
The results highlight the superiority of the **AHP method** over the **Non-AHP method** in all scenarios tested, demonstrating higher profitability while optimizing resource allocation.

| Method   | Scenario     | Demand | Freelancers | Profit ($)    | Demand without Freelancers | Freelancers without Demand |
|----------|--------------|--------|-------------|---------------|----------------------------|----------------------------|
| Non-AHP  | High (70%)   | 4,122  | 7,000       | 5,553,000.00  | 789                        | 3,667                      |
| AHP      | High (70%)   | 4,122  | 7,000       | 6,401,200.00  | 910                        | 3,788                      |
| **Diff** | High (70%)   | -      | -           | **848,200.00** | **121**                    | **121**                    |
| Non-AHP  | Medium (20%) | 4,122  | 2,000       | 3,156,920.00  | 2,412                      | 2,525                      |
| AHP      | Medium (20%) | 4,122  | 2,000       | 3,324,480.00  | 2,290                      | 2,403                      |
| **Diff** | Medium (20%) | -      | -           | **167,560.00** | **122**                    | **122**                    |
| Non-AHP  | Low (10%)    | 4,122  | 1,000       | 1,816,920.00  | 3,199                      | 3,232                      |
| AHP      | Low (10%)    | 4,122  | 1,000       | 1,929,800.00  | 3,110                      | 3,199                      |
| **Diff** | Low (10%)    | -      | -           | **112,880.00** | **89**                     | **33**                     |

### Key Insights:
- **Profitability**: AHP consistently achieved higher profits across all scenarios, with the largest margin of **$848,200


### Notes
 - To get the jobs data was utilized [SerpAPI](https://serpapi.com/), jobs search module. You can test that out using the data_collection_and_wrangler.py script. Just keep in mind that you will need a valid SerpAPI key (look into .env.example file).
 - In order to process the data a valid OpenAI key is needed. Also you need credits into your OpenAI account in order to use the trained models utilized in this research.