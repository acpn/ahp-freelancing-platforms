import ahpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def select_best_talent(matched_talents):
    # Extract the list of talent IDs
    talents = [str(talent['talent_id']) for talent in matched_talents]

    # If there is only one talent, return it directly
    if len(talents) == 1:
        return matched_talents[0]

    # Define the criteria
    criteria = ['hourly_rate', 'has_main_skill', 'talent_experience_years',
                'number_of_matched_skills', 'talent_english_level']

    # Define pairwise comparisons of criteria (you can adjust these values)
    decision_matrix = {
        ('hourly_rate', 'has_main_skill'): 3,
        ('hourly_rate', 'talent_experience_years'): 5,
        ('hourly_rate', 'number_of_matched_skills'): 7,
        ('hourly_rate', 'talent_english_level'): 9,
        ('has_main_skill', 'talent_experience_years'): 3,
        ('has_main_skill', 'number_of_matched_skills'): 5,
        ('has_main_skill', 'talent_english_level'): 7,
        ('talent_experience_years', 'number_of_matched_skills'): 3,
        ('talent_experience_years', 'talent_english_level'): 5,
        ('number_of_matched_skills', 'talent_english_level'): 3
    }

    criteria_pc = ahpy.Compare('Criteria', decision_matrix, precision=3,
                               random_index='saaty')
    
    criteria_weights = criteria_pc.local_weights

    # For each criterion, create pairwise comparisons among talents
    subcriteria = {}
    for criterion in criteria:
        comparisons = {}
        for i in range(len(talents)):
            for j in range(i+1, len(talents)):
                talent_i = matched_talents[i]
                talent_j = matched_talents[j]
                id_i = str(talent_i['talent_id'])
                id_j = str(talent_j['talent_id'])
                value_i = talent_i[criterion]
                value_j = talent_j[criterion]

                # Handle cost and benefit criteria
                if criterion == 'hourly_rate':  # Cost criterion
                    # Since lower is better, invert the values for comparison
                    if value_i != 0 and value_j != 0:
                        ratio = value_j / value_i
                    else:
                        ratio = 1
                else:  # Benefit criteria
                    if value_i != 0 and value_j != 0:
                        ratio = value_i / value_j
                    else:
                        ratio = 1
                comparisons[(id_i, id_j)] = ratio

        subcriteria[criterion] = ahpy.Compare(criterion, comparisons,
                                              precision=3,
                                              random_index='saaty')

    # print(f"Criteria Consistency Ratio: {criteria_pc.consistency_ratio}")
    # for criterion in criteria:
    #     if isinstance(subcriteria[criterion], ahpy.Compare):
    #         cr = subcriteria[criterion].consistency_ratio
    #         print(f"Consistency Ratio for {criterion}: {cr}")
    # Initialize overall priorities dictionary
    overall_priorities = {talent_id: 0 for talent_id in talents}

    # For each criterion, get the local priorities and
    # compute overall priorities
    for criterion in criteria:
        # Access weights directly
        local_priorities = subcriteria[criterion].local_weights  
        weight = criteria_weights[criterion]  # Weight of the criterion
        for talent_id in talents:
            overall_priorities[talent_id] += weight * local_priorities[
                talent_id]

    best_talent_id = max(overall_priorities, key=overall_priorities.get)

    best_talent = next(
        talent for talent in matched_talents if str(
            talent['talent_id']) == best_talent_id
    )

    return best_talent


def allocate(df_jobs: pd.DataFrame, 
             df_talents: pd.DataFrame,
             use_ahp: bool):
    allocations = []
    job_match_counts = []

    df_talents = df_talents.sort_values(by=['hourly_rate'])
    for job in df_jobs.itertuples():
        matches = []
        job_skills = set(skill.strip().lower()
                         for skill in job.parsed_skills.split(','))

        main_skill = job.main_skill

        for talent in df_talents.itertuples():

            in_price_range = talent.hourly_rate < job.job_rate
            has_experience = talent.experience_years >=\
                job.requested_experience

            talent_skills = set(skill.strip().lower()
                                for skill in talent.skills.split(','))

            has_main_skill = 1 if main_skill in talent_skills else 0

            matching_skills = job_skills & talent_skills

            number_of_matched_skills = len(matching_skills)

            if (in_price_range and has_experience and
                    number_of_matched_skills > 0):

                matches.append({
                    'talent_id': talent.id,
                    'hourly_rate': talent.hourly_rate,
                    # 'hourly_rate': (talent.hourly_rate
                    #                 if has_main_skill == 1
                    #                 else talent.hourly_rate + 10),
                    'talent_experience_years': talent.experience_years,
                    'number_of_matched_skills': number_of_matched_skills,
                    'talent_english_level': talent.english_level,
                    'matching_skills': ', '.join(matching_skills),
                    'talent_skills': talent.skills,
                    'talent_index': talent.Index,
                    'has_main_skill': has_main_skill
                })
                
                
                # Store the number of matches for this job
                job_match_counts.append({
                    'job_id': job.job_id,
                    'job_title': job.title,
                    'job_skills': job.parsed_skills,
                    'num_of_matches': len(matches)
                })
                
                
                if not use_ahp:
                    allocations.append({
                        'job_id': job.job_id,
                        'talent_id': talent.id,
                        'job_skills': job.parsed_skills,
                        'talent_skills': talent.skills,
                        'job_rate': job.job_rate,
                        'talent_hourly_rate': talent.hourly_rate,
                        'talent_english_level': talent.english_level,
                        'talent_experience_years': talent.experience_years,
                        'number_of_matched_skills': number_of_matched_skills,
                        'matching_skills': matching_skills,
                        'has_main_skill': has_main_skill
                    })

                    # Drop the allocated talent from df_talents
                    df_talents.drop(index=talent.Index, inplace=True)

                    # Drop the allocated job from df_jobs
                    df_jobs.drop(index=job.Index, inplace=True)
                    break

        if use_ahp:
            if matches:
                top_n = 15

                # In case we have more than 15 possibilities
                matches_filtered = matches[:top_n]

                best_talent = select_best_talent(matches_filtered)

                allocations.append({
                    'job_id': job.job_id,
                    'talent_id': best_talent['talent_id'],
                    'job_skills': job.parsed_skills,
                    'talent_skills': best_talent['talent_skills'],
                    'job_rate': job.job_rate,
                    'talent_hourly_rate': best_talent['hourly_rate'],
                    'talent_english_level': best_talent['talent_english_level'],
                    'talent_experience_years': best_talent[
                        'talent_experience_years'],
                    'number_of_matched_skills': best_talent[
                        'number_of_matched_skills'],
                    'matching_skills': best_talent['matching_skills']
                })

                # Drop the allocated talent from df_talents
                df_talents.drop(index=best_talent['talent_index'],
                                inplace=True)

                # Drop the allocated job from df_jobs
                df_jobs.drop(index=job.Index, inplace=True)
                
    # Create a DataFrame of job match counts
    df_job_match_counts = pd.DataFrame(job_match_counts)

    # Sort by the number of matched talents and take the top 10
    df_top_10_jobs = df_job_match_counts.sort_values(by='num_of_matches', ascending=False).head(50)
    print(df_top_10_jobs)

    return pd.DataFrame(allocations), df_jobs, df_talents


def calculate_profit(df_matches, hours_per_week=40):
    df_matches['profit_per_allocation'] = (
        df_matches['job_rate'] - df_matches['talent_hourly_rate'])\
            * hours_per_week

    total_profit = df_matches['profit_per_allocation'].sum()

    return total_profit


def stats(df_jobs: pd.DataFrame, df_talents: pd.DataFrame, jobs_frac: float,
          talents_frac: float, use_ahp):
    df_jobs_sample = df_jobs.sample(
        frac=jobs_frac, random_state=42).reset_index(drop=True)

    df_talents_sample = df_talents.sample(
        frac=talents_frac, random_state=42).reset_index(drop=True)
    
    num_jobs = len(df_jobs_sample)
    num_talents = len(df_talents_sample)

    df_matches, orphan_jobs, orphan_talents = allocate(df_jobs_sample,
                                                       df_talents_sample,
                                                       use_ahp)

    shortage = len(orphan_jobs)
    surplus = len(orphan_talents)
    profit = calculate_profit(df_matches)

    return profit, shortage, surplus, num_jobs, num_talents


def plot_results(df_results):
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Convert profits to millions for the profit chart
    df_results['Profit_M'] = df_results['Profit'] / 1e6

    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), gridspec_kw={'wspace': 0.3})

    # Set common style and font sizes
    title_font = {'fontsize': 18}
    label_font = {'fontsize': 14}

    # Profit Plot
    sns.barplot(
        x='Cenário', y='Profit_M', hue='Método', data=df_results, ax=axes[0]
    )
    axes[0].set_title('Comparação de Lucro', **title_font)
    axes[0].set_xlabel('Cenário', **label_font)
    axes[0].set_ylabel('Lucro (Milhões $)', **label_font)
    axes[0].legend(title='Método', fontsize=12, title_fontsize=14)

    # Add labels on top of the bars
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(
            f'{height:.2f}M',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=12
        )

    # Shortages Plot
    sns.barplot(
        x='Cenário', y='Shortages', hue='Método', data=df_results, ax=axes[1]
    )
    axes[1].set_title('Comparação de Demanda sem Freelancer', **title_font)
    axes[1].set_xlabel('Cenário', **label_font)
    axes[1].set_ylabel('Número de Demanda sem Freeelancer', **label_font)
    axes[1].legend(title='Método', fontsize=12, title_fontsize=14)

    # Add labels on top of the bars
    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(
            f'{int(height)}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=12
        )

    # Surplus Plot
    sns.barplot(
        x='Cenário', y='Surplus', hue='Método', data=df_results, ax=axes[2]
    )
    axes[2].set_title('Quantidade de Freelancers não utilizados', **title_font)
    axes[2].set_xlabel('Cenário', **label_font)
    axes[2].set_ylabel('Número de Freelancers não utilizados', **label_font)
    axes[2].legend(title='Método', fontsize=12, title_fontsize=14)

    # Add labels on top of the bars
    for p in axes[2].patches:
        height = p.get_height()
        axes[2].annotate(
            f'{int(height)}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=12
        )

    # Save the figure with higher DPI and JPG format for better quality
    plt.tight_layout()
    plt.savefig("comparisons.jpg", dpi=600, bbox_inches='tight', format='jpg')

    # Show the plot
    plt.show()


def main():
    df_jobs = pd.read_csv('jobs.csv')
    df_jobs = df_jobs.drop(columns=['Unnamed: 0']).reset_index()

    df_jobs['requested_experience'] = df_jobs[
        'requested_experience'].astype('int')

    # The first skill will be used as main skill
    df_jobs['main_skill'] = df_jobs['parsed_skills'].apply(
        lambda x: x.split(',')[0].strip().lower())
    
    df_jobs.describe()

    df_talents = pd.read_csv('mock_talents.csv')
    
    #stats(df_jobs, df_talents, 0.1, 0.1, True)
    
    # List to store results
    results = []

    # Scenarios and their corresponding job and talent fractions
    scenarios = [
        ('Alta Demanda 70%', 0.7),
        ('Média Demanda 20%', 0.2),
        ('Baixa Demanda 10%', 0.1)
    ]

    for scenario_name, fraction in scenarios:
        # AHP Allocation
        ahp_profit, ahp_shortages, ahp_surplus, num_jobs, num_talents = stats(
            df_jobs.copy(), df_talents.copy(), fraction, fraction, True)

        results.append({
            'Cenário': scenario_name,
            'Método': 'AHP',
            'Profit': ahp_profit,
            'Shortages': ahp_shortages,
            'Surplus': ahp_surplus,
            'Number of Jobs': num_jobs,
            'Number of Talents': num_talents
        })

        # Non-AHP Allocation
        profit, shortages, surplus, num_jobs, num_talents = stats(
            df_jobs.copy(), df_talents.copy(), fraction, fraction, False)

        results.append({
            'Cenário': scenario_name,
            'Método': 'NoN-AHP',
            'Profit': profit,
            'Shortages': shortages,
            'Surplus': surplus,
            'Number of Jobs': num_jobs,
            'Number of Talents': num_talents
        })

    # Create DataFrame from results
    df_results = pd.DataFrame(results)

    # Display the results DataFrame
    print("\nAllocation Results:")
    print(df_results)

    # Proceed to plotting
    plot_results(df_results)

main()