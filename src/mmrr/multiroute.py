"""Priority quota union: fusion, text, image; backfill duplicates within each route."""


def quota_union(routes, quotas):
    if len(routes) != len(quotas) or any(q < 0 for q in quotas):
        raise ValueError('Invalid route quotas')
    chosen, seen, sources = [], set(), []
    for route_index, (route, quota) in enumerate(zip(routes, quotas)):
        added = 0
        for item in route:
            if added == quota:
                break
            if item not in seen:
                chosen.append(item)
                seen.add(item)
                sources.append(route_index)
                added += 1
        if added != quota:
            raise ValueError('Route depth insufficient to fill unique quota')
    # Ordering only: unweighted RRF using available route ranks; does not select candidates.
    rank_maps = [{item: rank for rank, item in enumerate(route, 1)} for route in routes]
    scores = [sum(1 / (60 + ranks[item]) for ranks in rank_maps if item in ranks) for item in chosen]
    order = sorted(range(len(chosen)), key=lambda i: (-scores[i], chosen[i]))
    return [chosen[i] for i in order], [scores[i] for i in order], [sources[i] for i in order]
