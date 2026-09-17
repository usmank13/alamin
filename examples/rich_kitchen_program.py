scene(
    prompt="A 60 square metre working prep kitchen with a real retrieved refrigerator and countertop microwave, two stainless prep tables, a serving counter, four base cabinets, two drawer units, two wall cabinets, two dry-goods shelves, and varied containers, jars, bottles and trays. Keep appliance access clear and place all small items on support surfaces.",
    space=space(kind="kitchen",area_m2=60),
    objects=[
        place("cold_storage","fridge"),
        place("prep","prep_table",count=2,zone="prep"),
        place("service","counter",zone="service"),
        place("cabinet","base_cabinet",count=4,zone="storage"),
        place("drawers","drawer_unit",count=2,zone="prep"),
        place("upper","wall_cabinet",count=2,zone="storage"),
        place("shelving","shelf",count=2,zone="storage"),
        place("microwave","microwave",zone="prep"),
        place("dry_goods","container",count=12,zone="storage"),
        place("preserves","jar",count=8,zone="storage"),
        place("bottles","bottle",count=6,zone="prep"),
        place("trays","tray",count=3,zone="prep"),
    ],
)
