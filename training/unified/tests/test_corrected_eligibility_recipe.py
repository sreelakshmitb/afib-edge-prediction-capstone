from controlled_repair import (
    RECIPES,
    input_mode_for_recipe,
    lead_bins_for_epoch,
)


def test_corrected_eligibility_recipe_contract():
    recipe = "groupnorm-curriculum-corrected-eligibility"

    assert recipe in RECIPES
    assert input_mode_for_recipe(recipe) == "full"
    assert lead_bins_for_epoch(recipe, 0) == ("00_04", "04_08")
    assert len(lead_bins_for_epoch(recipe, 6)) == 5
