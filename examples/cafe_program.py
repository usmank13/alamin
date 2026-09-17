scene(
    prompt="A compact cafe kitchen with a prep table, three cabinets, two drawer units, a storage shelf, and eight containers.",
    space=space(kind="cafe", area_m2=36),
    objects=[
        place("prep", "prep_table"),
        place("cabinet", "base_cabinet", count=3),
        place("drawers", "drawer_unit", count=2),
        place("storage", "shelf"),
        place("goods", "container", count=8),
    ],
)
