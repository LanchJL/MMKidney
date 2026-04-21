# batch_valis_register_traverse_only_bigguard_v2.py
import os
import sys
from valis import registration, slide_io

# 根目录：每个子文件夹是一个病例（如 .../16S02292）
DATA_ROOT = "/media/a6000/3E5C99E35C9995ED/jcy/dataset/medical/WSIs_for_formal_experiment"
# 结果根目录：每个病例结果写到该目录下同名子目录
RESULTS_BASE = "/media/a6000/3E5C99E35C9995ED/jcy/ACMIL/ACMIL-main/reg_image_3"

# 触发“超大图降载”的阈值（任意一张 level-0 最大片边超过该值即触发）
BIG_SIDE_PX = 12000

# 你原来的默认参数（保持不变）
DEF_MAX_PROC = 1500
DEF_MAX_NONR = 2200
DEF_ALIGN_TO_REF = True
DEF_CROP = "reference"

# 大图降载时的替代参数（仅大图病例生效）
BIG_MAX_PROC = 1200
BIG_MAX_NONR = 1400
BIG_ALIGN_TO_REF = False
BIG_CROP = "overlap"

def list_mrxs(slide_dir: str):
    return sorted([f for f in os.listdir(slide_dir) if f.lower().endswith(".mrxs")])

def level0_max_side(slide_dir: str, fname: str) -> int:
    """返回单张 slide 的 level-0 最大边长度（px）。失败则返回 0。"""
    try:
        fpath = os.path.join(slide_dir, fname)
        reader_cls = slide_io.get_slide_reader(fpath, series=0)
        reader = reader_cls(fpath, series=0)
        sizes = reader.metadata.slide_dimensions  # [(w,h), (w,h), ...]，level 0 在索引 0
        if not sizes:
            return 0
        w0, h0 = sizes[0]
        return int(max(w0, h0))
    except Exception:
        return 0

def is_big_case(slide_dir: str, mrxs_list):
    """只要该病例下有任意一张 MRXS 的 level-0 最大边超过阈值，就视为大图病例。"""
    max_side_across = 0
    for fname in mrxs_list:
        ms = level0_max_side(slide_dir, fname)
        max_side_across = max(max_side_across, ms)
        # 小优化：遇到超阈值就不再继续
        if ms >= BIG_SIDE_PX:
            return True, max_side_across
    return (max_side_across >= BIG_SIDE_PX), max_side_across

def process_case(case_dir: str):
    case = os.path.basename(case_dir.rstrip("/"))

    slide_src_dir = case_dir
    results_dst_dir = os.path.join(RESULTS_BASE, case)
    registered_slide_dst_dir = results_dst_dir + "/registered_slides"

    # 参考图严格按“病例名-HE.mrxs”
    reference_img = f"{case}-HE.mrxs"
    ref_path = os.path.join(slide_src_dir, reference_img)
    if not os.path.exists(ref_path):
        print(f"[WARN] {case}: reference not found -> {reference_img} ; skip.")
        return

    mrxs_files = list_mrxs(slide_src_dir)
    if not mrxs_files:
        print(f"[WARN] {case}: no .mrxs under {slide_src_dir} ; skip.")
        return

    cases = ['15S19353', '15S27549' ,'15S28767']
    if case in cases:
        print(f"[WARN] {case}: 存在; skip.")
        return

    # —— 大图病例判定：检查“该病例下所有 .mrxs”的 level-0 尺寸 —— #
    big_case, max_side = is_big_case(slide_src_dir, mrxs_files)

    # 参数选择
    if big_case:
        max_proc     = BIG_MAX_PROC
        max_nonrigid = BIG_MAX_NONR
        align_to_ref = BIG_ALIGN_TO_REF
        crop_mode    = BIG_CROP
        micro_rigid_registrar_cls = None  # ⭐ 关键：禁用主流程里的高分辨率刚性重匹配
        do_micro     = False              # ⭐ 关键：跳过 register_micro，避免再上 level-0
    else:
        max_proc     = DEF_MAX_PROC
        max_nonrigid = DEF_MAX_NONR
        align_to_ref = DEF_ALIGN_TO_REF
        crop_mode    = DEF_CROP
        micro_rigid_registrar_cls = None  # 可设为默认（None 表示使用 VALIS 默认设置）
        do_micro     = True               # 保持你原来的流程

    print(f"\n=== {case} ===")
    print(f"[INFO] reference_img_f: {reference_img}")
    print(f"[INFO] max level-0 side across MRXS: {max_side}  (big_case={big_case})")
    if big_case:
        print(f"[INFO] BIG-GUARD → max_proc={max_proc}, max_nonrigid={max_nonrigid}, "
              f"micro_rigid_registrar_cls=None, register_micro={do_micro}, crop={crop_mode}")

    try:
        # —— 初始化（与原逻辑一致，仅在大图病例时多了 micro_rigid_registrar_cls=None）——
        registrar = registration.Valis(
            slide_src_dir,
            results_dst_dir,
            max_processed_image_dim_px=max_proc,
            max_non_rigid_registration_dim_px=max_nonrigid,
            create_masks=True,
            reference_img_f=reference_img,
            img_list=[os.path.join(slide_src_dir, f) for f in mrxs_files],

            # 关键两行：
            micro_rigid_registrar_cls=None,  # 关闭“高分辨率刚性重匹配”（A）
            crop_for_rigid_reg=False,  # 禁止刚性阶段的“放大到更高分辨率裁ROI”

            check_for_reflections=False,
        )

        # 主流程：刚性 + 非刚性
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()

        # 微注册：仅在非大图病例执行（保持你原来的细化步骤）
        # if do_micro:
        #     registrar.register_micro(
        #         max_non_rigid_registration_dim_px=max_nonrigid,
        #         align_to_reference=align_to_ref
        #     )

        # 保存（与原来一致；大图病例会用更保守的裁剪）
        registrar.warp_and_save_slides(
            registered_slide_dst_dir=registered_slide_dst_dir,
            crop=crop_mode,
            non_rigid=True,
            # 如确认是 uint8 并想进一步省 I/O，可启用：
            # compression="jpeg", Q=85,
        )
    finally:
        # 一定要杀 JVM
        try:
            registration.kill_jvm()
        except Exception:
            pass
def get_done_cases(results_base: str) -> set:
    """读取结果目录下所有子文件夹名，作为已完成病例集合。"""
    if not os.path.isdir(results_base):
        return set()
    return {
        d for d in os.listdir(results_base)
        if os.path.isdir(os.path.join(results_base, d))
    }

def main():
    if not os.path.isdir(DATA_ROOT):
        print(f"[ERR] DATA_ROOT missing: {DATA_ROOT}")
        sys.exit(1)

    # 1) 待处理病例列表（DATA_ROOT 下的子文件夹）
    cases = [
        os.path.join(DATA_ROOT, d)
        for d in sorted(os.listdir(DATA_ROOT))
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ]
    if not cases:
        print(f"[WARN] No case directories under {DATA_ROOT}")
        return

    # 2) 读取已经配准过的病例名（RESULTS_BASE 下的子文件夹）
    done_cases = get_done_cases(RESULTS_BASE)
    print(f"[INFO] 已存在结果的病例数量: {len(done_cases)}")

    os.makedirs(RESULTS_BASE, exist_ok=True)

    for case_dir in cases:
        case = os.path.basename(case_dir.rstrip("/"))
        # 3) 如果病例名已存在于结果目录，直接跳过
        if case in done_cases:
            print(f"[WARN] {case}: 已存在; skip.")
            continue

        try:
            process_case(case_dir)  # 你的原有处理函数
        except Exception as e:
            print(f"[ERROR] {case_dir}\n{e}")
            try:
                registration.kill_jvm()
            except Exception:
                pass

if __name__ == "__main__":
    main()




# 例：Docker（VALIS 官方镜像）
# docker run --rm -it \
#   --memory=64g --shm-size=16g \
#   -e PYTHONUNBUFFERED=1 \
#   -v /media/a6000:/media/a6000 \
#   cdgatenbee/valis-wsi:1.2.0 \
#   python3 /media/a6000/3E5C99E35C9995ED/jcy/KLM/batch_valis_register_thumbs.py

# docker run --rm -it --memory=64g \
#    -v /media/a6000:/media/a6000 \
#    cdgatenbee/valis-wsi:1.2.0 \
#    python3 /media/a6000/3E5C99E35C9995ED/jcy/KLM/batch_valis_register_thumbs.py

