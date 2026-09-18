scene(
    prompt="An 80 square metre warehouse and packing room with two stocked pallet racks, three empty Euro pallets staged against the walls, a parked pallet jack, two packing tables against the walls, three hinged tool-storage cabinets, two sliding parts drawers, and twelve small containers of packing supplies on support surfaces. Keep an entrance and circulation space clear. The pallet jack and loaded racks are static scenery, not lifting-task targets.",
    space=space(kind="warehouse", area_m2=80),
    objects=[
        place("rack", "pallet_rack", count=2, zone="storage", asset_request={"query": "ShelfE", "candidate_id": "warehouse:models/aws_robomaker_warehouse_ShelfE_01", "placement": "wall"}),
        place("pallet", "pallet", count=3, zone="staging", asset_request={"query": "euro pallet", "candidate_id": "gazebo:euro_pallet"}),
        place("jack", "pallet_jack", zone="staging", asset_request={"query": "PalletJackB", "candidate_id": "warehouse:models/aws_robomaker_warehouse_PalletJackB_01"}),
        place("packing", "prep_table", count=2, zone="packing"),
        place("tools", "base_cabinet", count=3, zone="storage"),
        place("parts", "drawer_unit", count=2, zone="packing"),
        place("supplies", "container", count=12, zone="packing"),
    ],
    relations=[relation("against_wall", ["packing", "pallet"], required=True)],
)
