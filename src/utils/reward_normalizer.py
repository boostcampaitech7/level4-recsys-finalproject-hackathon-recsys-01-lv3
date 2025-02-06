import numpy as np

class RewardNormalizer:
    def __init__(self):
        self.revenue_stats = {'mean': 0, 'var': 1, 'count': 0}
        self.rcr_change_stats = {'mean': 0, 'var': 1, 'count': 0}
        self.precision_stats = {'mean': 0, 'var': 1, 'count': 0}

    def update_stats(self, stats, value):
        stats['count'] += 1
        old_mean = stats['mean']
        stats['mean'] += (value - stats['mean']) / stats['count']
        stats['var'] += (value - old_mean) * (value - stats['mean'])

    def normalize(self, stats, value):
        std = (stats['var'] / stats['count']) ** 0.5 if stats['count'] > 0 else 1
        normalized_value = (value - stats['mean']) / (std + 1e-8)
        return normalized_value

    def normalize_reward(self, revenue, rcr_change, precision_at_k_value):
        # Update stats
        self.update_stats(self.revenue_stats, revenue)
        self.update_stats(self.rcr_change_stats, rcr_change)
        self.update_stats(self.precision_stats, precision_at_k_value)

        # Normalize values
        norm_revenue = self.normalize(self.revenue_stats, revenue)
        norm_rcr_change = self.normalize(self.rcr_change_stats, rcr_change)
        norm_precision = self.normalize(self.precision_stats, precision_at_k_value)

        # Combine normalized values into a single reward
        raw_reward = (
            0.4 * norm_revenue +
            0.3 * norm_rcr_change +
            0.3 * norm_precision
        )

        # Ensure reward is non-negative by shifting the minimum reward to zero
        return max(0, raw_reward)