template <class D, class M, class A>
class recommender_t
{
public:
	typedef D dataset_type;
	typedef M dataset_mask_type;
	typedef A algorithm_type;
	
	recommender_t(const dataset_type &dataset, 
				  const dataset_mask_type &dataset_mask, 
				  const algorithm_type &recommender_algorithm = algorithm_type())
		: m_dataset(dataset), m_dataset_mask(dataset_mask)
	{}
	dataset_type operator()(const dataset_type &request, 
							const dataset_mask_type &request_mask)
	{
		dataset_type prediction = m_recommender_algorithm(m_dataset, 
														  m_dataset_mask, 
														  request, 
														  request_mask);
		// for all rows in 'prediction'
		// std::sort in descending order
		return prediction;
	}
private:
	const dataset_type &m_dataset;
	const dataset_mask_type &m_dataset_mask;
	const algorithm_type &m_recommender_algorithm;
};

template <class D, class M, class A, class S>
class cross_validator_t
{
public:
	typedef D dataset_type;
	typedef M dataset_mask_type;
	typedef A algorithm_type;
	typedef S dataset_divider_type;
	
	cross_validator_t(const dataset_type &dataset, 
					  const dataset_mask_type &dataset_mask, 
					  const algorithm_type &recommender_algorithm = algorithm_type(), 
					  const dataset_divider_type &divider = dataset_divider_type())
		: m_dataset(dataset), m_dataset_mask(dataset_mask), m_divider(divider)
	{}
	
	void operator()()
	{
		
	}
	
private:
	const dataset_type &m_dataset;
	const dataset_mask_type &m_dataset_mask;
	const algorithm_type &m_recommender_algorithm;
	const dataset_divider_type &m_divider;
};
