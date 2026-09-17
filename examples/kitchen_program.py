scene(
    prompt="A commercial prep kitchen with two prep tables, three base cabinets, two drawer units, a dry-goods shelf, and twelve containers.",
    space=space(kind="kitchen", area_m2=60),
    objects=[
        place("prep", "prep_table", count=2),
        place("cabinet", "base_cabinet", count=3),
        place("drawers", "drawer_unit", count=2),
        place("storage", "shelf"),
        place("goods", "container", count=12),
    ],
)
